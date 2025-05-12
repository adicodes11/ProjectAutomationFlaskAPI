from flask import Flask, jsonify
from flask_cors import CORS

from app import analyze_project_bp
from chatbot import chatbot_bp
from chat_with_documents import chat_with_documents_bp
from task_assignment_automator import assign_tasks_bp

app = Flask(__name__)

# Improved CORS setup with more specific configuration
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, 
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"])

# Register Blueprints
app.register_blueprint(analyze_project_bp)           # For project analysis
app.register_blueprint(chatbot_bp)                   # For chatbot functionality
app.register_blueprint(chat_with_documents_bp)       # For chat-with-documents API
app.register_blueprint(assign_tasks_bp)              # For task assignment automation

# Global error handler for CORS preflight requests
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)