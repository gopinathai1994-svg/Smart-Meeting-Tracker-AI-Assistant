import os
import json
import smtplib
import traceback
from email.message import EmailMessage
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import whisper

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

from dotenv import load_dotenv
load_dotenv() 

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
app = Flask(__name__)
CORS(app)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

# 🔥 NEW: Securely fetch variables using os.getenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# Optional: Check if variables are loaded (For debugging)
if not all([MONGO_URI, GEMINI_API_KEY, SMTP_EMAIL, SMTP_PASSWORD]):
    print("⚠️ WARNING: Some environment variables are missing in .env file!")

try:
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✅ MongoDB Connected Successfully!")
except Exception as e:
    print(f"❌ MongoDB Connection Failed: {e}")

db = client["student"]
meetings_collection = db["meetings"]

print("Loading Whisper AI model...")
whisper_model = whisper.load_model("base")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=GEMINI_API_KEY)
    print("✅ Gemini AI Initialized!")
except Exception as e:
    print(f"❌ Gemini AI Failed: {e}")

# ==========================================
# 2. LANGGRAPH AI AGENTS DEFINITION
# ==========================================

# ADDED: meeting_title to State
class MeetingState(TypedDict):
    raw_transcript: str
    meeting_title: str  # <-- ADDED
    attendees: List[Dict[str, str]]
    summary: str
    action_items: List[Dict[str, str]]
    transcript_split: List[Dict[str, str]]
    email_drafts: Dict[str, str]

def extract_text_from_response(response):
    if isinstance(response.content, list):
        first_item = response.content[0]
        if isinstance(first_item, dict):
            return first_item.get('text', '')
        return str(first_item)
    return str(response.content)

def analyze_transcript_node(state: MeetingState):
    try:
        # 🔥 UPGRADED STRICT PROMPT
        prompt = f"""
        You are an expert meeting analyzer with 10+ years of experience.
        Project/Meeting Title: {state['meeting_title']}
        Attendees List: {state['attendees']}
        
        Transcript:
        {state['raw_transcript']}
        
        CRITICAL TASKS:
        1. Create a concise, professional summary (3-4 lines).
        2. Extract Action Items with:
           - Task: Clear, specific description
           - Assignee: MUST match EXACTLY from Attendees List (name only)
           - Deadline: Specific day/date (e.g., "Friday, October 25", "Next Monday", "November 1, 2024")
        3. Split transcript by speakers logically.
        
        OUTPUT FORMAT (Strictly JSON only, NO markdown, NO explanations):
        {{
          "summary": "Professional meeting summary here",
          "transcript_split": [{{"speaker": "Speaker Name", "text": "what they said"}}],
          "action_items": [
            {{"task": "Specific task description", "assignee": "Exact Name", "deadline": "Friday, October 25, 2024"}}
          ]
        }}
        """
        response = llm.invoke(prompt)
        try:
            raw_text = extract_text_from_response(response)
            clean_json = raw_text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return {
                "summary": data.get("summary", "No summary generated."),
                "action_items": data.get("action_items", []),
                "transcript_split": data.get("transcript_split", [])
            }
        except Exception as json_err:
            print(f"JSON Parse Error: {json_err}")
            return {"summary": "Error parsing summary", "action_items": [], "transcript_split": []}
    except Exception as e:
        print(f"Analyze Node Error: {e}")
        return {"summary": "Error in analysis", "action_items": [], "transcript_split": []}

def draft_emails_node(state: MeetingState):
    try:
        drafts = {}
        attendees_map = {att['name']: att['email'] for att in state['attendees']}
        meeting_title = state.get('meeting_title', 'Our Project')
        
        tasks_by_person = {}
        for item in state['action_items']:
            assignee = item['assignee']
            if assignee not in tasks_by_person:
                tasks_by_person[assignee] = []
            tasks_by_person[assignee].append(item)

        for person, tasks in tasks_by_person.items():
            if person in attendees_map:
                # 🔥 HIGHLY PROFESSIONAL EMAIL PROMPT
                prompt = f"""
                You are a Senior Project Manager writing action item emails.
                
                Project Name: {meeting_title}
                Recipient: {person}
                
                Action Items for {person}:
                {json.dumps(tasks, indent=2)}
                
                EMAIL REQUIREMENTS:
                1. SUBJECT LINE: Must include project name and be action-oriented
                   Example: "Action Items: {meeting_title} - Your Tasks & Deadlines"
                
                2. GREETING: Professional (e.g., "Dear {person},")
                
                3. BODY STRUCTURE:
                   - Opening: Acknowledge the meeting briefly
                   - Main Content: List ONLY {person}'s tasks using bullet points
                   - Each bullet must show: Task + Specific Deadline
                   - Make deadlines VERY clear and prominent
                   - Do NOT mention other people's tasks
                   - Closing: Professional, supportive tone
                
                4. FORMAT EXAMPLE:
                   Dear [Name],
                   
                   Following our meeting about [Project], here are your action items:
                   
                   • Task: [Description]
                     Deadline: [Specific Day/Date]
                   
                   • Task: [Description]
                     Deadline: [Specific Day/Date]
                   
                   Please let me know if you need any support.
                   Best regards,
                   Project Management Team
                
                5. OUTPUT: Strictly JSON format: {{"subject": "...", "body": "..."}}
                """
                response = llm.invoke(prompt)
                try:
                    raw_text = extract_text_from_response(response)
                    clean_json = raw_text.strip().replace("```json", "").replace("```", "").strip()
                    email_data = json.loads(clean_json)
                    full_email = f"Subject: {email_data['subject']}\n\n{email_data['body']}"
                    drafts[attendees_map[person]] = full_email
                except Exception as e:
                    print(f"Email draft error for {person}: {e}")
                    drafts[attendees_map[person]] = "Error generating email draft."
                    
        return {"email_drafts": drafts}
    except Exception as e:
        print(f"Draft Node Error: {e}")
        return {"email_drafts": {}}

# Build LangGraph Workflow
workflow = StateGraph(MeetingState)
workflow.add_node("analyze", analyze_transcript_node)
workflow.add_node("draft", draft_emails_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "draft")
workflow.add_edge("draft", END)
app_graph = workflow.compile()

# ==========================================
# 3. EMAIL SENDING FUNCTION
# ==========================================
def send_strict_emails(email_drafts):
    sent_status = {}
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        
        for recipient_email, email_content in email_drafts.items():
            msg = EmailMessage()
            lines = email_content.split('\n')
            subject = lines[0].replace("Subject: ", "").strip()
            body = "\n".join(lines[2:])
            
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = SMTP_EMAIL
            msg['To'] = recipient_email
            
            server.send_message(msg)
            sent_status[recipient_email] = "Success"
            print(f"✅ Strictly sent email to {recipient_email}")
        server.quit()
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        for email in email_drafts:
            sent_status[email] = f"Failed: {str(e)}"
    return sent_status

# ==========================================
# 4. FLASK ROUTES
# ==========================================
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/process_meeting', methods=['POST'])
def process_meeting():
    try:
        print("=== NEW REQUEST RECEIVED ===")
        
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
            
        meeting_title = request.form.get('title', 'Untitled Meeting')
        attendees_json = request.form.get('attendees', '[]')
        
        try:
            attendees = json.loads(attendees_json)
        except json.JSONDecodeError:
            return jsonify({"success": False, "error": "Invalid attendees format"}), 400
        
        print(f"Title: {meeting_title}")
        print(f"Attendees: {attendees}")
        print(f"Audio File: {audio_file.filename}")
        
        file_path = os.path.join("uploads", audio_file.filename)
        audio_file.save(file_path)
        print(f"File saved to: {file_path}")
        
        print("Transcribing audio...")
        result = whisper_model.transcribe(file_path)
        raw_transcript = result["text"]
        print(f"Transcript snippet: {raw_transcript[:100]}...")
        
        print("Running AI Agents...")
        initial_state = {
            "raw_transcript": raw_transcript,
            "meeting_title": meeting_title,  # <-- ADDED
            "attendees": attendees,
            "summary": "",
            "action_items": [],
            "transcript_split": [],
            "email_drafts": {}
        }
        final_state = app_graph.invoke(initial_state)
        print("AI Agents completed!")
        
        print("Sending emails...")
        email_status = send_strict_emails(final_state.get('email_drafts', {}))
        
        meeting_record = {
            "title": meeting_title,
            "attendees": attendees,
            "summary": final_state.get('summary', ''),
            "transcript_split": final_state.get('transcript_split', []),
            "action_items": final_state.get('action_items', []),
            "email_drafts": final_state.get('email_drafts', {}),
            "email_status": email_status
        }
        
        print("Saving to MongoDB...")
        db_result = meetings_collection.insert_one(meeting_record)
        meeting_record["_id"] = str(db_result.inserted_id)
        
        print("✅ Saved to MongoDB successfully!")
        
        return jsonify({
            "success": True,
            "data": meeting_record
        })
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

# if __name__ == '__main__':
#     print("🚀 Starting Flask Server...")
#     app.run(debug=True, port=5000)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)