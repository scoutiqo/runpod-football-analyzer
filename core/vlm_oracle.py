import base64
import json
import os
import requests

API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ask_oracle(frame_path):
    if not API_KEY: return "unknown"

    b64 = encode_image(frame_path)

    # THE SCOUTIQO ONTOLOGY
    prompt = """
    You are a Senior Football Analyst. Analyze this frame and classify the action into ONE of these 60 labels:
    
    1. ATTACK: "short_pass", "long_ball", "cross", "cutback", "through_ball", "chip_pass", "shot_foot", "shot_header", "volley", "penalty_shot"
    2. CONTROL: "ball_carry", "dribble_attempt", "dribble_success", "dribble_failure", "first_touch", "ball_recovery"
    3. DEFENSE: "tackle", "sliding_tackle", "interception", "block_shot", "block_pass", "clearance", "pressing_action", "defensive_error"
    4. DUEL: "aerial_duel", "ground_duel", "shielding"
    5. STOPPAGE: "corner_taken", "throw_in", "goal_kick", "free_kick_cross", "free_kick_short", "penalty_awarded", "offside", "foul_committed", "goal", "kickoff"
    6. KEEPER: "save", "catch_claim", "punch", "keeper_sweep", "keeper_throw", "keeper_kick"
    
    If nothing significant is happening, return "none".
    Reply ONLY JSON: {"label": "your_label"}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ],
        "max_tokens": 60,
        "response_format": { "type": "json_object" }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        content = json.loads(result['choices'][0]['message']['content'])
        return content.get("label", "unknown")
    except:
        return "unknown"
