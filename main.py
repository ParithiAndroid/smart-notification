# main.py - Hybrid Agent (Code Logic + AI Creativity)
import os
import json
import time
import random
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# --- CONFIGURATION ---
# Get Key: https://aistudio.google.com/
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- DATA MODELS ---
class PaymentInfo(BaseModel):
    count: int
    courseId: int
    courseName: str
    lastVisitedAt: int

class VideoInfo(BaseModel):
    completion: int
    lastWatchedAt: int
    videoId: int
    videoName: str

class LearnerRequest(BaseModel):
    payments: Dict[str, PaymentInfo]
    videos: Dict[str, VideoInfo]

# --- 1. PYTHON LOGIC TOOL (Reliable) ---
# This runs locally. It NEVER hallucinates because it is pure code.
def analyze_learner_data(data: LearnerRequest) -> List[Dict]:
    events = []
    
    # Logic A: Payments = High Priority Promotion
    # If 'payments' map is not empty, user is interested in buying.
    if data.payments:
        # Taking the first payment entry for context
        first_key = next(iter(data.payments))
        p_data = data.payments[first_key]
        events.append({
            "event_type": "payment_promotion",
            "context": f"User visited payment page for {p_data.courseName} {p_data.count} times.",
            "recommended_action": "Offer a discount or nudge to complete purchase.",
            "priority": "Urgent"
        })

    # Logic B: Videos
    for vid_id, vid in data.videos.items():
        if 1 <= vid.completion < 95:
            events.append({
                "event_type": "video_resume",
                "context": f"Stopped watching {vid.videoName} at {vid.completion}%.",
                "priority": "High"
            })
        elif vid.completion >= 95:
             events.append({
                "event_type": "video_milestone",
                "context": f"Finished watching {vid.videoName}.",
                "priority": "Medium"
            })
        elif vid.completion == 0:
             events.append({
                "event_type": "video_start",
                "context": f"Has not started {vid.videoName} yet.",
                "priority": "Low"
            })
            
    return events

# --- 2. API ENDPOINT ---
@app.post("/generate_notifications")
async def generate_notifications(data: LearnerRequest):
    # Step 1: Extract Signals using Python (Fast & Accurate)
    derived_events = analyze_learner_data(data)
    
    if not derived_events:
        return [] # No notifications needed

    payload_str = json.dumps(derived_events)
    
    # Step 2: Use AI for CREATIVITY only (Text Generation)
    # We pass the *events*, not the raw data. This saves tokens and improves quality.
    system_instruction = """
   You are a Creative Copywriter for an EdTech App.

INPUT:
A list of specific 'events' derived from learner activity.

GOAL:
Generate high-converting, modern, catchy push notifications
based ONLY on the provided events.

EVENT GUIDELINES:
- "video_resume": Encouraging. “Keep going!”, “You’re so close.”
- "video_milestone": Celebratory. “Great job!”, “You did it!”
- "payment_promotion": Urgent/Exciting. “Unlock full access”, “Don’t miss out.”
- "video_start": Curiosity. “Check this out.”

STYLE GUIDE:
- Modern, concise, punchy (Twitter/X style).
- Encouraging tone with light urgency.
- Max 1 emoji per title.
- No repetition across notifications.

OUTPUT RULES:
- Create **1 to 3 notifications total**, even if events are many.
- Pick the **highest-priority** or **most impactful** events.
- Return ONLY a valid JSON array following the schema provided.
- No Markdown. No extra text.
    """
    
    # Retry Logic for 429/5xx errors
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Events to Notify: {payload_str}",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type='application/json',
                    response_schema={
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "STRING"},
                                "title": {"type": "STRING"},
                                "body": {"type": "STRING"},
                                "sendNow": {"type": "BOOLEAN"},
                                "type": {
                                    "type": "STRING", 
                                    "enum": ["milestone", "reminder", "alert", "promotion", "engagement"]
                                }
                            },
                            "required": ["id", "title", "body", "sendNow", "type"]
                        }
                    }
                )
            )
            return json.loads(response.text)

        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            time.sleep((base_delay * (2 ** attempt)) + random.uniform(0, 0.5))
