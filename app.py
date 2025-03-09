# app.py
from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MongoDB configuration: using your TaskFlowNet database
MONGO_URI = os.getenv("MONGO_URI")
client_mongo = MongoClient(MONGO_URI)
db = client_mongo["TaskFlowNet"]

# Existing projects collection
projects_collection = db["projects"]
# Collections for storing analysis documents separately
analysis_collection = db["analysis"]
raw_collection = db["rawAnalysis"]

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def analyze_project_data_with_gemini(project_data):
    """
    Converts project_data into a prompt and calls the Google Gemini 2.0 Flash API
    to generate detailed project management insights in JSON format.
    """
    # Convert the project data (a dictionary) into a formatted JSON string.
    project_details = json.dumps(project_data, indent=2)
    
    # Build a prompt instructing Gemini to return output in JSON format.
    prompt = (
        "Given the following project details:\n\n"
        f"{project_details}\n\n"
        "Please provide detailed project management insights and recommendations "
        "as a valid JSON object with the following keys:\n"
        "- suggestedTime: string (e.g., '10 weeks')\n"
        "- suggestedBudget: number (e.g., 15000)\n"
        "- riskAssessment: string\n"
        "- recommendedTeamStructure: an object with keys like 'lead', 'developers', 'QA'\n"
        "- memberRecommendations: an object where each key is a team member name and the value is an object with recommendations\n"
    )
    
    # Call the Gemini API to generate content based on the prompt.
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    # Return the raw text response from Gemini.
    return response.text

@app.route('/api/analyze_project', methods=['POST'])
def analyze_project():
    """
    API endpoint to analyze project data.
    Accepts a JSON payload with project details (which must include an "_id" field),
    uses Google Gemini to generate recommendations, and then creates two new documents:
      - One in the 'analysis' collection with the structured analysis.
      - One in the 'rawAnalysis' collection with the raw text output.
    Both documents include a "projectId" field (the ObjectId from the projects collection).
    """
    try:
        project_data = request.get_json()
        if not project_data:
            return jsonify({"message": "No project data provided"}), 400
        
        if "_id" not in project_data:
            return jsonify({"message": "Project _id is required to link analysis data."}), 400
        
        project_id = project_data["_id"]
        
        # Get the raw analysis text from Gemini.
        raw_analysis = analyze_project_data_with_gemini(project_data)
        
        # Attempt to parse the raw analysis into a JSON object.
        try:
            processed_analysis = json.loads(raw_analysis)
        except Exception as parse_err:
            processed_analysis = {"error": "Parsing failed", "raw": raw_analysis}
        
        # Create a document for the structured analysis.
        analysis_doc = {
            "projectId": ObjectId(project_id),
            "analysis": processed_analysis,
            "analysisTimestamp": datetime.utcnow()
        }
        analysis_insert_result = analysis_collection.insert_one(analysis_doc)
        
        # Create a document for the raw analysis.
        raw_doc = {
            "projectId": ObjectId(project_id),
            "rawAnalysis": raw_analysis,
            "rawTimestamp": datetime.utcnow()
        }
        raw_insert_result = raw_collection.insert_one(raw_doc)
        
        return jsonify({
            "message": "Project analysis completed successfully",
            "analysis_id": str(analysis_insert_result.inserted_id),
            "raw_analysis_id": str(raw_insert_result.inserted_id),
            "analysis": processed_analysis
        }), 200

    except Exception as e:
        print("Error analyzing project:", e)
        return jsonify({"message": "Internal Server Error"}), 500

if __name__ == '__main__':
    # Use Waitress to serve the app in production.
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
