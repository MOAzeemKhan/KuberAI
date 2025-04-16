# app.py - Main Flask application for KuberAI financial assistant
print("STEP 1: Starting app.py")
from flask import Flask, jsonify, render_template, request, session

print("STEP 2: Flask imported")

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
import uuid
from datetime import datetime
import threading  # Make sure this import is here
from threading import Thread  # This was missing - explicitly import Thread
import time

# LangChain-related imports
from langchain_community.vectorstores import Chroma

print("STEP 4: Chroma imported")

from langchain_community.embeddings import HuggingFaceEmbeddings

print("STEP 5: Embeddings imported")

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Table extractor dependencies
import shutil
from pathlib import Path

from together import Together
# Load LLM & retriever
from dotenv import load_dotenv

load_dotenv()
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
def ask_llama3_70b(prompt):
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    response = together_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def get_together_response(prompt):
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# App configuration
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')
processing_status = {}  # Dictionary to track processing status by ID


# Global variable to store latest extracted table context
latest_table_context = ""


os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
persist_directory = "docs/chroma_rag/"
os.makedirs(persist_directory, exist_ok=True)
if not os.path.exists(persist_directory):
    chroma_db = Chroma.from_documents(
        documents=load_docs(),
        embedding=embeddings,
        persist_directory=persist_directory
    )
    chroma_db.persist()

try:
    chroma_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="economic_data"
    )
    retriever = chroma_db.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    print(f"Warning: Could not load existing Chroma DB: {e}")
    # Initialize with empty DB
    chroma_db = Chroma.from_documents(
        documents=[],
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="economic_data"
    )
    retriever = chroma_db.as_retriever(search_kwargs={"k": 4})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are KuberAI, a friendly and knowledgeable financial assistant. Your job is to help users understand finance.
Use the context below to answer the user's question. Be factual, concise, and explain in simple terms.

Context:
{context}

Question:
{question}

Answer:
"""
)

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 0.1, "max_new_tokens": 512})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False,
                                       chain_type_kwargs={"prompt": prompt})


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/index")
def dashboard():
    return render_template("index.html")


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files['file']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = str(uuid.uuid4())
    filepath = f"uploads/input_{timestamp}.pdf"

    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    file.save(filepath)

    # Initialize processing status
    processing_status[task_id] = {
        "status": "processing",
        "message": "PDF upload received. Processing...",
        "timestamp": timestamp,
        "filename": file.filename
    }

    def run_extraction():
        print(f"[THREAD] 🟡 Entered thread. Attempting to import and call extractor...")
        try:
            from extractor import extract_tables
            print(f"[THREAD] Running extract_tables for {pdf_path}")
            json_path = extract_tables(filepath, timestamp)
            global latest_table_context
            print("Running Ma Boy")
            if json_path and os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    latest_table_context = "\n".join([f"{k}: {v}" for k, v in data.items()])

                # Update status to complete
                processing_status[task_id]["status"] = "complete"
                processing_status[task_id]["message"] = "PDF processing complete. You can now ask questions about it."
                processing_status[task_id]["completion_time"] = time.time()
                print("[THREAD] Extraction successful, context ready.")
            else:
                # Update status to error
                print("[THREAD] JSON path not found or empty.")
                processing_status[task_id]["status"] = "error"
                processing_status[task_id]["message"] = "Failed to process PDF. No data extracted."
                processing_status[task_id]["completion_time"] = time.time()
        except Exception as e:
            print(f"Extraction error: {str(e)}")
            processing_status[task_id]["status"] = "error"
            processing_status[task_id]["message"] = f"Error processing PDF: {str(e)}"
            processing_status[task_id]["completion_time"] = time.time()

    Thread(target=run_extraction).start()  # This line was causing the error
    print(f"Started extraction thread for task ID: {task_id}")
    return jsonify({
        "status": "processing",
        "message": "PDF uploaded. Extracting now...",
        "task_id": task_id
    })


@app.route("/check_processing/<task_id>", methods=["GET"])
def check_processing(task_id):
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    return jsonify({"status": "not_found", "message": "Processing task not found"}), 404


@app.route("/api/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"status": "error", "message": "Empty question."}), 400

        greetings = ["hi", "hello", "hey", "yo", "what's up"]
        if question.lower() in greetings:
            return jsonify({
                "status": "success",
                "answer": "Hey there! I'm KuberAI, your financial assistant. Ask me about stocks, investing, or the economy 💹"
            })

        # Build context if PDF was uploaded
        # STEP 1: Get Chroma-based RAG context
        rag_docs = retriever.get_relevant_documents(question)
        rag_context = "\n".join([doc.page_content for doc in rag_docs])

        # STEP 2: Combine with PDF-extracted table context
        if latest_table_context:
            combined_context = latest_table_context + "\n\n" + rag_context
        else:
            combined_context = rag_context

        # STEP 3: Inject into prompt
        full_prompt = prompt.format(context=combined_context, question=question)

        # 🔥 Ask Together AI's LLaMA 3.3 70B
        response = get_together_response(full_prompt)
        answer = response.strip()

        # ✂️ Clean any boilerplate
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        answer = answer.replace("KuberAI:", "").strip()

        return jsonify({"status": "success", "answer": answer})

    except Exception as e:
        print(f"Error in QA chain: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "I had trouble processing that question. Please try rephrasing."
        }), 500



# Clean up old processing statuses periodically
def cleanup_processing_status():
    while True:
        time.sleep(3600)  # Clean up every hour
        current_time = time.time()
        to_remove = []
        for task_id, status_info in processing_status.items():
            # Remove completed tasks older than 1 hour
            if status_info["status"] in ["complete", "error"] and \
                    current_time - status_info.get("completion_time", current_time) > 3600:
                to_remove.append(task_id)

        for task_id in to_remove:
            del processing_status[task_id]


# Start cleanup thread when app starts - using Thread class correctly here
threading.Thread(target=cleanup_processing_status, daemon=True).start()

if __name__ == "__main__":
    print("🧠 KuberAI backend is starting...")
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("docs/chroma_rag", exist_ok=True)
    app.run(debug=True)