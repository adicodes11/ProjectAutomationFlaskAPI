from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(".env.local")
from google import genai
from flask_cors import CORS
import PyPDF2

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
client_mongo = MongoClient(MONGO_URI)
db = client_mongo["ProjectAutomation"]

# Collections
projects_collection = db["projects"]
analysis_collection = db["analysis"]
raw_collection = db["rawAnalysis"]
conversation_collection = db["chatWithDocuments"]  # A new collection for doc-based chats

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

############################################################
# In-memory storage for document text (if you want to store
# it only for the session rather than in DB).
############################################################
uploaded_documents = {}  # { session_id or userEmail or something : "document text" }

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
    project = projects_collection.find_one({"_id": ObjectId(project_id)})
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

@app.route("/api/chat_with_documents", methods=["POST"])
def chat_with_documents():
    """
    Single endpoint to handle:
      1) Document upload (if file is in request.files)
      2) Chat with the document (if JSON body with 'query', 'projectId', 'userEmail')
    We store the doc text in either in-memory or in the DB. For demonstration, we store in memory.

    - If a file is uploaded, we do not require 'query', 'projectId', 'userEmail'.
    - If no file is present, we assume it's a chat request with JSON body:
         { projectId, userEmail, query, conversationId(optional) }
    """
    try:
        # 1) Check if a file is uploaded (multipart/form-data)
        if "file" in request.files:
            file = request.files["file"]
            filename = file.filename.lower()
            if not filename:
                return jsonify({"message": "No file selected"}), 400

            # Parse the file
            if filename.endswith(".pdf"):
                document_text = extract_text_from_pdf(file)
            elif filename.endswith(".txt"):
                document_text = file.read().decode("utf-8", errors="ignore")
            else:
                return jsonify({"message": "Unsupported file type (only PDF or TXT)"}), 400

            # Optionally store the text in memory or DB
            # For demonstration, let's just store in memory with key "global"
            uploaded_documents["global"] = document_text

            return jsonify({
                "message": f"File '{filename}' processed successfully!",
                "documentLength": len(document_text)
            }), 200

        # 2) Otherwise, we assume it's a JSON-based chat request
        data = request.get_json()
        if not data:
            return jsonify({"message": "No JSON data provided"}), 400

        project_id = data.get("projectId")
        user_email = data.get("userEmail")
        query = data.get("query")
        conversation_id = data.get("conversationId")

        if not project_id or not user_email or not query:
            return jsonify({"message": "projectId, userEmail, and query are required"}), 400

        # Fetch project context
        context = fetch_project_context(project_id)

        # Also fetch the uploaded doc text from memory (if any)
        doc_text = uploaded_documents.get("global", "")
        if doc_text:
            context += "\n\nDocument Content:\n" + doc_text

        # Construct prompt
        prompt = (
            f"Project Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            "Answer:"
        )

        # Call Gemini
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw_answer = response.text
        answer = clean_response(raw_answer)

        # Prepare conversation log
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

        if conversation_id:
            # If we have an existing conversation, update
            conv_id = ObjectId(conversation_id)
            conversation_collection.update_one(
                {"_id": conv_id},
                {"$push": {"messages": {"$each": [query_entry, answer_entry]}}}
            )
        else:
            # Create a new conversation doc
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

if __name__ == "__main__":
    from waitress import serve
    print("ChatWithDocuments service running on port 8082")
    serve(app, host="0.0.0.0", port=8082)
