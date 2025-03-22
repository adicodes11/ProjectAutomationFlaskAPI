# app.py
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

# MongoDB configuration: using your ProjectAutomation (or ProjectAutomation) database
MONGO_URI = os.getenv("MONGO_URI")
client_mongo = MongoClient(MONGO_URI)
db = client_mongo["ProjectAutomation"]  # Change this name if your database is named differently

# Collections for storing documents
analysis_collection = db["analysis"]
raw_collection = db["rawAnalysis"]

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def generate_long_response(project_data):
    """
    1st AI call: Produce a long, multi-page analysis with no strict JSON constraints.
    """
    project_details = json.dumps(project_data, indent=2)
    prompt = (
        "You are an expert project management advisor.\n"
        "Please provide a very detailed, multi-page analysis of this project. "
        "Discuss scope, budget, timeline, risk factors, team structure, phases, potential pitfalls, advanced ideas, and anything relevant.\n\n"
        f"Project details:\n{project_details}\n\n"
        "Feel free to be as thorough as possible. No strict format is required for this step."
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def extract_json_from_text(text):
    """
    Attempt to extract a JSON object from text using regex.
    Returns a dictionary if successful, else None.
    """
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except Exception as e:
            print("Error parsing extracted JSON:", e)
            return None
    return None

def parse_into_structured_json(raw_text):
    """
    2nd AI call: Transform the raw text into a rich JSON object with many key–value pairs.
    If direct parsing fails, attempts regex extraction as a fallback.
    """
    parse_prompt = (
        "You are an assistant that converts long text into a rich JSON structure.\n"
        "Given the raw text below, extract and create a JSON object with the following keys exactly:\n"
        "  - suggestedTime (string)\n"
        "  - suggestedBudget (number)\n"
        "  - riskAssessment (string)\n"
        "  - recommendedTeamStructure (object)\n"
        "  - memberRecommendations (object)\n"
        "  - phases (array or object)\n"
        "  - potentialRisks (array)\n"
        "  - riskMitigation (array or object)\n"
        "  - advancedIdeas (array or object)\n"
        "  - sdlcMethodology (string)\n"
        "If any field is missing, return an empty string or null for that field.\n"
        "Return ONLY valid JSON. Do not include any extra text, markdown, or disclaimers.\n\n"
        f"Raw text:\n{raw_text}\n"
    )
    parse_response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=parse_prompt,
    )
    structured_text = parse_response.text
    print("Structured text from Gemini:", structured_text)

    # Try direct parsing first.
    try:
        structured_data = json.loads(structured_text)
        if isinstance(structured_data, dict):
            return structured_data
    except Exception as e:
        print("Direct JSON parsing failed:", e)

    # Fallback: extract JSON block using regex.
    extracted = extract_json_from_text(structured_text)
    if extracted:
        return extracted

    return {"error": "Second-pass parsing failed", "raw": structured_text}

@app.route("/api/analyze_project", methods=["POST"])
def analyze_project():
    """
    Single route that:
      1. Takes project data (with _id),
      2. Calls Gemini for a long raw analysis response,
      3. Stores that raw response in the rawAnalysis collection,
      4. Calls Gemini again to parse the raw text into a structured JSON,
      5. Stores the final structured JSON (as key–value pairs) in the analysis collection,
      6. Returns document references and the structured analysis.
    """
    try:
        project_data = request.get_json()
        if not project_data:
            return jsonify({"message": "No project data provided"}), 400

        if "_id" not in project_data:
            return jsonify({"message": "Project _id is required to link analysis data."}), 400

        project_id = project_data["_id"]
        print("Received project data for ID:", project_id)

        # Step 1: Generate long raw analysis.
        raw_analysis = generate_long_response(project_data)
        print("Raw analysis generated.")

        # Step 2: Store the raw response.
        raw_doc = {
            "projectId": ObjectId(project_id),
            "rawAnalysis": raw_analysis,
            "createdAt": datetime.utcnow()
        }
        raw_result = raw_collection.insert_one(raw_doc)
        print("Raw analysis document inserted with ID:", raw_result.inserted_id)

        # Step 3: Transform raw analysis into a structured JSON.
        structured_data = parse_into_structured_json(raw_analysis)
        print("Structured data parsed:", structured_data)

        # Step 4: Store the structured analysis.
        analysis_doc = {
            "projectId": ObjectId(project_id),
            "analysis": structured_data,  # Stored as individual fields in MongoDB document
            "analysisTimestamp": datetime.utcnow()
        }
        analysis_result = analysis_collection.insert_one(analysis_doc)
        print("Structured analysis document inserted with ID:", analysis_result.inserted_id)

        # Step 5: Return document references and structured analysis.
        return jsonify({
            "message": "Project analysis completed successfully",
            "raw_analysis_id": str(raw_result.inserted_id),
            "analysis_id": str(analysis_result.inserted_id),
            "analysis": structured_data
        }), 200

    except Exception as e:
        print("Error analyzing project:", e)
        return jsonify({"message": "Internal Server Error"}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
