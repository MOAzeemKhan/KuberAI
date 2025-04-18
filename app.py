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
import threading
from threading import Thread
import time
import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
from typing import List, Tuple, Any, Dict
import logging

# LangChain-related imports
from langchain_community.vectorstores import Chroma

print("STEP 4: Chroma imported")

from langchain_community.embeddings import HuggingFaceEmbeddings

print("STEP 5: Embeddings imported")

from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Table extractor dependencies
import shutil
from pathlib import Path

from together import Together
# Load LLM & retriever
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

from intent_classifier import IntentClassifier

# Initialize the classifier with your pre-trained model
intent_classifier = IntentClassifier()

# Path to your pre-trained model file
model_path = os.path.join("kuberai_intent_classifier.pkl")

# Load the pre-trained model
intent_classifier.load_model(model_path)
print("Pre-trained intent classifier loaded successfully")


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


# TableExtractor class from the notebook
class TableExtractor:
    def __init__(self, pdf_path):
        '''self.huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            huggingfacehub_api_token=self.huggingfacehub_api_token,
            repo_id=self.repo_id,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 3000}
        )'''
        self.pdf_path = pdf_path
        pass

    def _image_list_(self, pdf_path: str) -> List[str]:
        try:
            images = convert_from_path(self.pdf_path)
            img_list = []
            for i, image in enumerate(images):
                image_name = f'temp_images/page_{i}.jpg'
                os.makedirs('temp_images', exist_ok=True)
                image.save(image_name, 'JPEG')
                img_list.append(image_name)
            return img_list
        except Exception as e:
            logging.error(f"Error converting PDF to images: {e}")
            raise

    def _preprocess_image_(self, image_path: str) -> Any:
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        except Exception as e:
            logging.error("Error during preprocessing", exc_info=True)
            raise

    def _detect_tables_(self, image: Any) -> List[Tuple[int, int, int, int]]:
        try:
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image.shape[0] / 30)))
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[1] / 30), 1))
            vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
            table_grid = cv2.add(horizontal_lines, vertical_lines)
            contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tables = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > image.size * 0.001:
                    tables.append((x, y, w, h))
            logging.info(f"Detected {len(tables)} tables.")
            return tables
        except Exception as e:
            logging.error("Error detecting tables", exc_info=True)
            raise

    def _extract_text_from_tables_(self, image: Any, tables: List[Tuple[int, int, int, int]]) -> List[str]:
        try:
            texts = []
            for (x, y, w, h) in tables:
                table_image = image[y:y + h, x:x + w]
                text = pytesseract.image_to_string(table_image, lang='eng')
                texts.append(text)
            logging.info(f"Extracted text from {len(tables)} tables.")
            return texts
        except Exception as e:
            logging.error("Error extracting text", exc_info=True)
            raise

    def extract_tables_and_text(self) -> List[str]:
        try:
            images = self._image_list_(self.pdf_path)
            all_tables_text = []
            for image_path in images:
                preprocessed_image = self._preprocess_image_(image_path)
                tables = self._detect_tables_(preprocessed_image)
                texts = self._extract_text_from_tables_(preprocessed_image, tables)
                all_tables_text.extend(texts)
            return all_tables_text
        except Exception as e:
            logging.error("Error extracting tables and text", exc_info=True)
            raise

    def extracted_data(self) -> List[str]:
        try:
            tables_text = self.extract_tables_and_text()
            answer = []
            for text in tables_text:
                cleaned_string = re.sub(r'[ \t]+', ' ', text)
                cleaned_string = re.sub(r'\n\s*\n', '', cleaned_string)
                answer.append(cleaned_string)
            return answer
        except Exception as e:
            logging.error("Error cleaning extracted data", exc_info=True)
            raise

    def response(self, content: str) -> str:
        try:
            '''template = """[INST]you are json formatter. your task analyze the given data {data} and must return answer as json. key doesn't have value return empty string. only generate json for given data's. all answers should be in json format (for all data).[/INST]"""
            prompt = PromptTemplate(template=template, input_variables=["data"])
            llm_chain = LLMChain(prompt=prompt, verbose=True, llm=self.llm)
            result = llm_chain.run({"data": content})
            return result'''
            prompt = f"""[INST]You are a JSON formatter. Your task is to analyze the given data and return a JSON output. 
            Key without value should return an empty string. Only generate JSON for the given data. 
            All answers must be in valid JSON format.

            DATA:
            {content}
            [/INST]"""

            response = get_together_response(prompt)
            return response

        except Exception as e:
            logging.error("Error during LLM response", exc_info=True)
            raise

    def safe_response_with_retry(self, content: str, retries=3, wait_time=10) -> str:
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"Attempt {attempt}/{retries}")
                return self.response(content)
            except Exception as e:
                logging.warning(f"⚠️ Attempt {attempt} failed. Retrying in {wait_time}s...\nError: {e}")
                time.sleep(wait_time)
        raise RuntimeError("❌ All retries failed for LLM inference.")

    def list_of_answer(self) -> List[str]:
        try:
            answer = self.extracted_data()
            final = []
            for i in range(len(answer)):
                logging.info(f"Processing table {i + 1}/{len(answer)}")
                result = self.safe_response_with_retry(answer[i])
                final.append(result)
            logging.info("Completed processing list of answers.")
            return final
        except Exception as e:
            logging.error("Error in list_of_answer", exc_info=True)
            raise

    def extract_and_combine_json(self, text_list: List[str]) -> Dict[str, Any]:
        try:
            pattern = r'```json\n({.*?})\n```'
            combined_json = {}
            for text in text_list:
                json_strings = re.findall(pattern, text, re.DOTALL)
                for json_str in json_strings:
                    try:
                        json_obj = json.loads(json_str)
                        # Merge with combined_json
                        combined_json.update(json_obj)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to decode JSON: {e}")
                        # Try without the pattern match if it fails
                        try:
                            json_obj = json.loads(text)
                            combined_json.update(json_obj)
                        except:
                            logging.warning("Failed to decode even without pattern matching")
            return combined_json
        except Exception as e:
            logging.error(f"Error in extract_and_combine_json: {e}", exc_info=True)
            return {}

    def key_value_pair(self) -> Dict[str, Any]:
        try:
            list_of_text = self.list_of_answer()
            combined_json = self.extract_and_combine_json(list_of_text)
            logging.info("Successfully combined JSON objects.")
            return combined_json
        except Exception as e:
            logging.error(f"Error in key_value_pair: {e}", exc_info=True)
            return {}


# Function to extract tables and save as JSON
def extract_tables(pdf_path, timestamp):
    logging.info(f"Starting table extraction for {pdf_path}")
    try:
        # Create output directory
        output_dir = os.path.join("static", "extracted")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize extractor
        extractor = TableExtractor(pdf_path)

        # Extract table data
        combined_json = extractor.key_value_pair()

        # Save to JSON file
        json_path = os.path.join(output_dir, f"extracted_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(combined_json, f, indent=2)

        logging.info(f"Table extraction complete. Results saved to {json_path}")

        # Clean up temp images
        if os.path.exists("temp_images"):
            shutil.rmtree("temp_images")

        return json_path
    except Exception as e:
        logging.error(f"Table extraction failed: {e}", exc_info=True)
        return None


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "✅ KuberAI is Running Successfully!"


# App configuration
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')
processing_status = {}  # Dictionary to track processing status by ID

# Global variable to store latest extracted table context
latest_table_context = ""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
persist_directory = "docs/chroma_rag/"
os.makedirs(persist_directory, exist_ok=True)

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
You are KuberAI, a friendly and knowledgeable financial assistant. 
Your job is to help users understand finance concepts in simple terms.

- If the user asks for a concept (like "equity financing"), explain **generally** without mentioning specific companies like Microsoft or Apple unless directly asked.
- If the user asks about a specific company, then you can use examples.
- Keep answers factual, simple, and to the point.

Context:
{context}

Question:
{question}

Answer:
"""
)

llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                     model_kwargs={"temperature": 0.1, "max_new_tokens": 512})
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
        try:
            # Call our extract_tables function
            logging.info(f"Running extract_tables for {filepath}")
            json_path = extract_tables(filepath, timestamp)
            global latest_table_context

            if json_path and os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data = json.load(f)
                    latest_table_context = "\n".join([f"{k}: {v}" for k, v in data.items()])

                # Update status to complete
                processing_status[task_id]["status"] = "complete"
                processing_status[task_id]["message"] = "PDF processing complete. You can now ask questions about it."
                processing_status[task_id]["completion_time"] = time.time()
                logging.info("Extraction successful, context ready.")
            else:
                # Update status to error
                logging.error("JSON path not found or empty.")
                processing_status[task_id]["status"] = "error"
                processing_status[task_id]["message"] = "Failed to process PDF. No data extracted."
                processing_status[task_id]["completion_time"] = time.time()
        except Exception as e:
            logging.error(f"Extraction error: {str(e)}")
            processing_status[task_id]["status"] = "error"
            processing_status[task_id]["message"] = f"Error processing PDF: {str(e)}"
            processing_status[task_id]["completion_time"] = time.time()

    Thread(target=run_extraction).start()
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


from flask import request, jsonify
import re

@app.route("/api/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"status": "error", "message": "Empty question."}), 400

        # Classify the intent using the pre-trained model
        intent, confidence = intent_classifier.classify(question)
        print(f"Detected intent: {intent} (confidence: {confidence:.2f})")
        
        # Apply confidence threshold for more reliable predictions
        # If confidence is too low, fall back to a more general approach
        if confidence < 0.6:
            # For low confidence predictions, use keyword detection as fallback
            clean_question = re.sub(r'[^\w\s]', '', question.lower())
            
            # Check for finance keywords
            finance_keywords = ["stock", "mutual fund", "bond", "crypto", "investment", 
                               "portfolio", "equity", "debt", "etf", "finance", "returns", 
                               "risk", "dividend", "interest", "compound", "market"]
            
            # Check for document keywords
            document_keywords = ["document", "pdf", "report", "upload", "table", "file", 
                                "extract", "statement"]
            
            if any(keyword in clean_question for keyword in finance_keywords):
                intent = "FINANCE_QUESTION"
                print("Low confidence - falling back to FINANCE_QUESTION based on keywords")
            elif any(keyword in clean_question for keyword in document_keywords):
                intent = "DOCUMENT_QUESTION"
                print("Low confidence - falling back to DOCUMENT_QUESTION based on keywords")
            # Other fallbacks can be added here
        
        # Handle based on intent
        if intent == "GREETING":
            return jsonify({
                "status": "success",
                "answer": "Hey there! I'm KuberAI, your financial assistant. Ask me about stocks, investing, finance, or upload a document! 📑"
            })
            
        elif intent == "HELP":
            return jsonify({
                "status": "success",
                "answer": "I'm KuberAI, your financial assistant. Here's how you can use me:\n\n"
                "• Ask me any finance-related questions\n"
                "• Upload financial documents (like PDFs) to get insights\n"
                "• Ask questions about data in uploaded documents\n"
                "• Learn about investment concepts, terminology, and strategies"
            })
            
        elif intent == "OFF_TOPIC":
            return jsonify({
                "status": "success",
                "answer": "I'm specialized in finance and investments 📈. Could you please ask me something related to stocks, portfolios, crypto, or financial planning?"
            })
            
        elif intent == "DOCUMENT_QUESTION":
            if latest_table_context:
                # Use the document context with the retriever
                rag_docs = retriever.get_relevant_documents(question)
                rag_context = "\n".join([doc.page_content for doc in rag_docs])
                combined_context = latest_table_context + "\n\n" + rag_context
                
                full_prompt = prompt.format(context=combined_context, question=question)
                response = get_together_response(full_prompt)
                answer = response.strip()
                
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                if "Question:" in answer:
                    answer = answer.split("Question:")[0].strip()
                answer = answer.replace("KuberAI:", "").strip()
                
                return jsonify({"status": "success", "answer": answer})
            else:
                return jsonify({
                    "status": "success",
                    "answer": "I don't see any uploaded documents. Would you like to upload a financial document for me to analyze?"
                })
                
        elif intent == "FINANCE_QUESTION":
            # Standard finance question handling with RAG
            rag_docs = retriever.get_relevant_documents(question)
            rag_context = "\n".join([doc.page_content for doc in rag_docs])
            
            full_prompt = prompt.format(context=rag_context, question=question)
            response = get_together_response(full_prompt)
            answer = response.strip()
            
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            if "Question:" in answer:
                answer = answer.split("Question:")[0].strip()
            answer = answer.replace("KuberAI:", "").strip()
            
            return jsonify({"status": "success", "answer": answer})

    except Exception as e:
        print(f"Error processing question: {str(e)}")
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
    os.makedirs("static/extracted", exist_ok=True)
    os.makedirs("temp_images", exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))