# chatbot.py
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

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB configuration: using your ProjectAutomation database
MONGO_URI = os.getenv("MONGO_URI")
client_mongo = MongoClient(MONGO_URI)
db = client_mongo["ProjectAutomation"]

# Collections for storing documents
projects_collection = db["projects"]
analysis_collection = db["analysis"]
raw_collection = db["rawAnalysis"]
conversation_collection = db["chatbotConversation"]

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def fetch_project_context(project_id):
    """
    Fetch and merge data from the projects, analysis, and rawAnalysis collections.
    Returns a combined context string for the project.
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
        context_parts.append("Structured Analysis:\n" + json.dumps(analysis.get("analysis"), default=str, indent=2))
    
    raw_analysis = raw_collection.find_one(
        {"projectId": ObjectId(project_id)},
        sort=[("createdAt", -1)]
    )
    if raw_analysis and raw_analysis.get("rawAnalysis"):
        context_parts.append("Raw Analysis:\n" + raw_analysis.get("rawAnalysis"))
    
    return "\n\n".join(context_parts)

def clean_response(text):
    """
    Cleans the raw response text from Gemini.
    1. Remove disclaimers (if any).
    2. Remove triple backticks and code fences.
    3. Remove double-asterisk markdown (**) that might clutter text.
    4. Remove repeated asterisks.
    5. Strip leading/trailing whitespace.
    """
    # 1) Remove disclaimers (common pattern: "disclaimer: ...")
    cleaned = re.sub(r"(disclaimer:.*?)(?=\n|$)", "", text, flags=re.IGNORECASE)

    # 2) Remove triple backticks and code fences
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)  # block fences
    cleaned = re.sub(r"`+", "", cleaned)              # single or triple backticks

    # 3) Remove double-asterisks (often used for bold in Markdown)
    cleaned = re.sub(r"\*\*", "", cleaned)

    # 4) Remove leftover repeated asterisks if they appear
    #    (this is optional—only do this if you're sure you don't want bullet asterisks)
    cleaned = re.sub(r"\*{2,}", "", cleaned)

    # 5) Trim whitespace
    cleaned = cleaned.strip()

    return cleaned

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    """
    Processes a user query using project context.
    Expected JSON payload:
      - projectId (string)
      - userEmail (string)
      - query (string)
      - conversationId (optional string)
    
    Merges context from projects, analysis, and rawAnalysis collections,
    constructs a prompt, calls Gemini for a response, cleans the response,
    stores the conversation, and returns the answer.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"message": "No data provided"}), 400
        
        project_id = data.get("projectId")
        user_email = data.get("userEmail")
        query = data.get("query")
        conversation_id = data.get("conversationId")  # Optional

        if not project_id or not user_email or not query:
            return jsonify({"message": "projectId, userEmail, and query are required"}), 400

        # Fetch combined project context.
        context = fetch_project_context(project_id)

        # Construct a stricter prompt to guide formatting:
        prompt = f"""
You are a helpful AI assistant that responds in a clean, concise, well-formatted text.
Please do not include triple backticks or disclaimers. 
Use headings, bullet points, or short paragraphs as needed. 
Avoid excessive asterisks or markdown fences.

Project Context:
{context}

User Query:
{query}

Answer:
"""

        # Call Gemini API.
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        # Clean the answer to remove any extraneous delimiters, disclaimers, etc.
        raw_answer = response.text
        answer = clean_response(raw_answer)

        # Prepare conversation entries.
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

        # Save conversation history.
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
        print("Error processing chatbot query:", e)
        return jsonify({"message": "Internal Server Error"}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8081)
