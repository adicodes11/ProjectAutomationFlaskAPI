# assign_tasks_module.py

from flask import Blueprint, request, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import json
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv(".env.local")
from google import genai
from flask_cors import CORS  # ✅ Added CORS import

assign_tasks_bp = Blueprint("assign_tasks_bp", __name__)
CORS(assign_tasks_bp, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)  # ✅ Updated CORS with specific origin

# MongoDB configuration: using your database (e.g., "ProjectAutomation")
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["ProjectAutomation"]

# Collections
projects_collection = db["projects"]
analysis_collection = db["analysis"]
raw_collection = db["rawAnalysis"]
team_assignments_collection = db["teamAssignments"]

# Google Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def generate_long_response(project_data):
    """Call Gemini API to generate a detailed long analysis text."""
    project_details = json.dumps(project_data, indent=2)
    prompt = (
        "You are an expert project management advisor.\n"
        "Provide a very detailed, multi-page analysis of the following project. "
        "Discuss scope, budget, timeline, risk factors, team structure, phases, potential pitfalls, advanced ideas, "
        "and any other relevant aspects.\n\n"
        f"Project details:\n{project_details}\n\n"
        "Be as thorough as possible. No strict format is required."
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text

def extract_json_from_text(text):
    """Try to extract a JSON object from the provided text using regex."""
    # Remove markdown code block markers if present
    cleaned_text = re.sub(r'```(json)?|```', '', text)
    
    # Try to extract JSON object
    match = re.search(r'(\{.*\})', cleaned_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except Exception as e:
            print("Error parsing extracted JSON:", e)
    return None

def parse_into_structured_json(raw_text):
    """
    Call Gemini API to transform the raw text into a structured JSON.
    If direct parsing fails, use regex extraction as fallback.
    """
    parse_prompt = (
        "You are an assistant that converts long text into a rich JSON structure.\n"
        "Extract and create a JSON object with the following keys exactly:\n"
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
        "If any field is missing, set it as empty or null.\n"
        "Return ONLY valid JSON without extra text.\n\n"
        f"Raw text:\n{raw_text}\n"
    )
    parse_response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=parse_prompt,
    )
    structured_text = parse_response.text
    print("Structured text from Gemini:", structured_text)
    try:
        # Remove markdown code block markers if present
        cleaned_text = re.sub(r'```(json)?|```', '', structured_text)
        structured_data = json.loads(cleaned_text)
        if isinstance(structured_data, dict):
            return structured_data
    except Exception as e:
        print("Direct JSON parsing failed:", e)
    extracted = extract_json_from_text(structured_text)
    if extracted:
        return extracted
    return {"error": "Second-pass parsing failed", "raw": structured_text}

def get_combined_context(project_id):
    """Retrieve and merge details from the project, analysis, and rawAnalysis collections."""
    context_parts = []
    project = projects_collection.find_one({"_id": ObjectId(project_id)})
    if project:
        proj_copy = dict(project)
        proj_copy.pop("_id", None)
        context_parts.append("Project Details:\n" + json.dumps(proj_copy, default=str, indent=2))
    analysis = analysis_collection.find_one({"projectId": ObjectId(project_id)}, sort=[("analysisTimestamp", -1)])
    if analysis and analysis.get("analysis"):
        context_parts.append("Structured Analysis:\n" + json.dumps(analysis.get("analysis"), default=str, indent=2))
    raw_analysis = raw_collection.find_one({"projectId": ObjectId(project_id)}, sort=[("createdAt", -1)])
    if raw_analysis and raw_analysis.get("rawAnalysis"):
        context_parts.append("Raw Analysis:\n" + raw_analysis.get("rawAnalysis"))
    return "\n\n".join(context_parts)

def generate_assignment_with_gemini(project_id, confirmed_team, start_date=None, total_days=None):
    """
    Use Gemini to generate a detailed assignment for each confirmed team member.
    The prompt includes the combined context, confirmed team details, and timeline info (if available).
    Expected JSON output format:
    {
      "assignments": {
        "email1": {
          "teamMemberName": "Name",
          "role": "Role",
          "tasks": [
             {"description": "Task description", "deadline": "YYYY-MM-DD", "status": "Pending", "progress": 0}
          ]
        },
        "email2": { ... }
      }
    }
    """
    combined_context = get_combined_context(project_id)
    prompt = (
        "You are an expert project management advisor. Based on the following project context and confirmed team details, "
        "generate a detailed task assignment plan for each team member in JSON format. For each team member, include their email, name, role, and an array of tasks. "
        "Each task should have a 'description', a 'deadline' (YYYY-MM-DD format), a 'status' (set to 'Pending'), 'progress' (0), and 'assignedAt' timestamp in ISO format. "
        "Distribute deadlines evenly over the project's timeline if timeline information is provided. "
        "If timeline information is not provided, leave deadlines as null.\n\n"
        "Project Context:\n" + combined_context + "\n\n"
        "Confirmed Team Details:\n" + json.dumps(confirmed_team, indent=2) + "\n\n"
    )
    if start_date and total_days:
        prompt += (
            f"Project Timeline: Start date is {start_date.isoformat()} and total project duration is {total_days} days.\n\n"
        )
    prompt += (
        "Generate a JSON object with an 'assignments' key mapping each team member's email to their assignment details. "
        "Return ONLY the valid JSON without any markdown code block markers (like ```json or ```) or other text."
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    generated_text = response.text
    print("Generated assignment text from Gemini:", generated_text)
    
    # Remove any markdown code block markers
    cleaned_text = re.sub(r'```(json)?|```', '', generated_text)
    
    try:
        assignment_data = json.loads(cleaned_text)
        if "assignments" in assignment_data:
            return assignment_data["assignments"]
    except Exception as e:
        print("Error parsing generated assignment JSON:", e)
    
    extracted = extract_json_from_text(generated_text)
    if extracted and "assignments" in extracted:
        return extracted["assignments"]
    return {}

# Fix route path to match what's used in the frontend
@assign_tasks_bp.route("/assign_tasks", methods=["POST", "OPTIONS"])
def assign_tasks():
    """
    Endpoint to assign tasks to confirmed team members for a project.
    Expects a JSON payload with:
      - projectId (string)
      - confirmedTeam (array of team member objects with required fields)
    This endpoint:
      1. Retrieves combined context from project, analysis, and rawAnalysis collections.
      2. Uses the Gemini API to generate a detailed assignment plan for each confirmed team member.
      3. Distributes deadlines evenly if timeline information is available.
      4. Upserts each team member's assignment document in the teamassignments collection.
    """
    # Handle preflight OPTIONS request explicitly
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"message": "No data provided"}), 400
        project_id = data.get("projectId")
        confirmed_team = data.get("confirmedTeam")
        if not project_id or not confirmed_team:
            return jsonify({"message": "Project ID and confirmed team details are required"}), 400

        # Fetch project to obtain timeline information (if available)
        project = projects_collection.find_one({"_id": ObjectId(project_id)})
        start_date = None
        total_days = None
        if project and project.get("timeline"):
            try:
                timeline_weeks = int(project.get("timeline"))
                total_days = timeline_weeks * 7
                start_date = project.get("createdAt")
                # Ensure start_date is a datetime object:
                if not isinstance(start_date, datetime):
                    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception as ex:
                print("Error parsing timeline:", ex)

        # Use Gemini to generate a detailed assignment plan.
        assignments = generate_assignment_with_gemini(project_id, confirmed_team, start_date, total_days)
        print("Final Assignments from Gemini:", assignments)

        # Upsert each team member's assignment document in the teamassignments collection.
        for email, assign in assignments.items():
            team_assignments_collection.update_one(
                {"email": email, "projectId": ObjectId(project_id)},
                {"$set": {
                    "teamMemberName": assign.get("teamMemberName", ""),
                    "role": assign.get("role", ""),
                    "tasks": assign.get("tasks", []),
                    "updatedAt": datetime.utcnow()
                }},
                upsert=True
            )

        response = jsonify({
            "message": "Tasks assigned successfully",
            "assignments": assignments
        })
        return response, 200

    except Exception as e:
        print("Error in assign_tasks:", e)
        return jsonify({"message": "Internal Server Error", "error": str(e)}), 500

# Keep the original route as well for backward compatibility
@assign_tasks_bp.route("/assignTasks", methods=["POST", "OPTIONS"])
def assign_tasks_original():
    return assign_tasks()