from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(".env.local")
from google import genai
import PyPDF2

chat_with_documents_bp = Blueprint('chat_with_documents', __name__)
from flask_cors import CORS
CORS(chat_with_documents_bp)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
client_mongo = MongoClient(MONGO_URI)
db = client_mongo["ProjectAutomation"]

# Collections
projects_collection = db["projects"]
analysis_collection = db["analysis"]
raw_collection = db["rawAnalysis"]
conversation_collection = db["chatWithDocuments"]  # For doc-based chats

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# In-memory storage for document text
uploaded_documents = {}  # Example: {"global": "document text"}

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file using PyPDF2."""
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()

def fetch_project_context(project_id):
    """
    Fetch and merge context from projects, analysis, and rawAnalysis.
    Returns a combined context string.
    """
    context_parts = []
    try:
        project = projects_collection.find_one({"_id": ObjectId(project_id)})
    except Exception as e:
        project = None

    if project:
        proj_copy = dict(project)
        proj_copy.pop("_id", None)
        context_parts.append("Project Details:\n" + json.dumps(proj_copy, default=str, indent=2))
    
    analysis = analysis_collection.find_one(
        {"projectId": ObjectId(project_id)},
        sort=[("analysisTimestamp", -1)]
    )
    if analysis and analysis.get("analysis"):
        context_parts.append(
            "Structured Analysis:\n" + json.dumps(analysis.get("analysis"), default=str, indent=2)
        )
    
    raw_analysis = raw_collection.find_one(
        {"projectId": ObjectId(project_id)},
        sort=[("createdAt", -1)]
    )
    if raw_analysis and raw_analysis.get("rawAnalysis"):
        context_parts.append("Raw Analysis:\n" + raw_analysis.get("rawAnalysis"))
    
    return "\n\n".join(context_parts)

def clean_response(text):
    """
    Cleans the raw response text from Gemini:
    - Removes code fences, markdown, extra asterisks, etc.
    """
    cleaned = re.sub(r"```[\s\S]*?\n", "", text)  # remove code fence blocks
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = re.sub(r"\*+", "", cleaned)         # remove extra asterisks
    return cleaned.strip()

@chat_with_documents_bp.route("/chat_with_documents", methods=["POST"])
def chat_with_documents():
    """
    Handles:
      1) Document upload (if a file is sent via multipart/form-data)
      2) Chat with the document (if a JSON body with 'query' and 'userEmail' is provided)
      
    If a projectId is not provided in the JSON payload, a new one is generated.
    """
    try:
        # 1) If a file is uploaded, process it
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename.lower()
            if not filename:
                return jsonify({"message": "No file selected"}), 400

            if filename.endswith(".pdf"):
                document_text = extract_text_from_pdf(file)
            elif filename.endswith(".txt"):
                document_text = file.read().decode("utf-8", errors="ignore")
            else:
                return jsonify({"message": "Unsupported file type (only PDF or TXT)"}), 400

            # For demonstration, store in memory
            uploaded_documents["global"] = document_text

            return jsonify({
                "message": f"File '{filename}' processed successfully!",
                "documentLength": len(document_text)
            }), 200

        # 2) Otherwise, process JSON-based chat request
        data = request.get_json()
        if not data:
            return jsonify({"message": "No JSON data provided"}), 400

        # If no projectId is provided, generate a new one.
        project_id = data.get("projectId") or str(ObjectId())
        user_email = data.get("userEmail")
        query = data.get("query")
        conversation_id = data.get("conversationId")

        if not user_email or not query:
            return jsonify({"message": "userEmail and query are required"}), 400

        # Fetch project context if available (empty if project not found)
        context = fetch_project_context(project_id)
        
        # Append document text if available
        doc_text = uploaded_documents.get("global", "")
        if doc_text:
            context += "\n\nDocument Content:\n" + doc_text

        # Construct prompt for Gemini
        prompt = (
            f"Project Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Answer:"
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw_answer = response.text
        answer = clean_response(raw_answer)

        # Prepare conversation log entries
        query_entry = {
            "timestamp": datetime.utcnow(),
            "role": "user",
            "message": query
        }
        answer_entry = {
            "timestamp": datetime.utcnow(),
            "role": "assistant",
            "message": answer
        }

        # If conversationId exists, update; otherwise, create a new conversation document.
        if conversation_id:
            conv_id = ObjectId(conversation_id)
            conversation_collection.update_one(
                {"_id": conv_id},
                {"$push": {"messages": {"$each": [query_entry, answer_entry]}}}
            )
        else:
            conversation_doc = {
                "projectId": ObjectId(project_id),
                "userEmail": user_email,
                "documentContent": doc_text,
                "messages": [query_entry, answer_entry],
                "createdAt": datetime.utcnow()
            }
            conv_result = conversation_collection.insert_one(conversation_doc)
            conversation_id = str(conv_result.inserted_id)

        return jsonify({
            "message": "Query processed successfully",
            "answer": answer,
            "conversationId": conversation_id
        }), 200

    except Exception as e:
        print("Error in chat_with_documents:", e)
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500
