# main.py - The Brain of your Agent
# This single file handles the request and calls Gemini

import os
import json
import time
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# SECURITY: 
# Get Key: https://aistudio.google.com/
# Set 'GEMINI_API_KEY' in Vercel > Settings > Environment Variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    payments: dict[str, PaymentInfo]
    videos: dict[str, VideoInfo]

@app.post("/generate_notifications")
async def generate_notifications(data: LearnerRequest):
    payload_str = json.dumps(data.model_dump())
    
    # Enhanced System Instruction for Catchy & Strict Output
    system_instruction = """
    You are an expert EdTech Notification Agent.
    
    Goal: Analyze learner activity to generate high-converting, modern, and catchy push notifications.
    
    Style Guide:
    - **Modern**: Use concise, punchy language. Think "Twitter/X" style short copy.
    - **Classic**: Use proven engagement hooks like "Don't break the chain" or "You're so close".
    - **Tone**: Encouraging, slightly urgent, but never annoying. Use emojis effectively (max 1 per title).
    
    Logic Rules:
    1. **Resume (High Priority)**: If video completion is > 0% and < 100%, create a 'reminder'. Focus on "Finishing what you started".
    2. **Celebration (Medium Priority)**: If video completion is 100%, create a 'milestone'. Make them feel great.
    3. **Discovery (Low Priority)**: If completion is 0%, create an 'engagement' hook to start the content.
    4. **Urgency**: If completion is 90-99%, set 'sendNow' to true.

    Output Requirement:
    - You MUST return a JSON Array based on the schema provided. 
    - No Markdown formatting.
    """
    
    # Retry Configuration for Robustness
    max_retries = 3
    base_delay = 1.0 # seconds
    retry_codes = ["429", "500", "503", "504"]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"Learner Data: {payload_str}",
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
            error_msg = str(e)
            is_last_attempt = attempt == max_retries - 1
            
            # Check if error is likely transient (Rate limit or Server error)
            is_transient = any(code in error_msg for code in retry_codes)

            if is_transient and not is_last_attempt:
                # Exponential Backoff + Jitter
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                print(f"Attempt {attempt + 1} failed with {error_msg}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                # If not transient or it's the last attempt, fail hard
                print(f"Request failed: {error_msg}")
                raise HTTPException(status_code=500, detail=str(e))
