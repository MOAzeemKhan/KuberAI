# --- app.py (with improved retriever & prompt) ---
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.config import INDICATORS
import itertools
import json

# LangChain-related imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load QA model once
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
persist_directory = "docs/chroma_rag/"
chroma_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="economic_data"
)
retriever = chroma_db.as_retriever(search_kwargs={"k": 4})  # Increased to k=4

template = """
You are KuberAI, a friendly and knowledgeable financial assistant. Your job is to help users understand finance.
Use only the context below to answer the user's question. Be factual, concise, and explain in simple terms.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False, chain_type_kwargs={"prompt": PROMPT})

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/index")
def dashboard():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"status": "error", "message": "Empty question."}), 400

        greetings = ["hi", "hello", "hey", "yo", "what's up"]
        if question.lower().strip() in greetings:
            return jsonify({"status": "success", "answer": "Hey there! I’m KuberAI, your financial assistant. Ask me about stocks, investing, or the economy 💹"})

        response = qa_chain.run(question)
        answer = response.strip().split("Answer:")[-1].strip()
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/evaluate", methods=["GET"])
def evaluate_model():
    try:
        with open("test_set.json") as f:
            test_cases = json.load(f)

        results = []
        correct_count = 0

        for case in test_cases:
            question = case["question"]
            expected = case.get("expected", "")
            expected_keywords = case.get("expected_keywords", [])

            response = qa_chain.run(question)
            answer = response.strip().split("Answer:")[-1].strip().lower()

            if expected_keywords:
                is_correct = all(k.lower() in answer for k in expected_keywords)
            elif expected:
                is_correct = any(k.lower() in answer for k in expected.split())
            else:
                is_correct = False

            results.append({
                "question": question,
                "expected": expected or ", ".join(expected_keywords),
                "answer": answer,
                "correct": is_correct
            })

            if is_correct:
                correct_count += 1

        accuracy = round((correct_count / len(results)) * 100, 2)
        return jsonify({"status": "success", "accuracy": accuracy, "results": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
