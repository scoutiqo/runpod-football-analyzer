import base64
import json
import os
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY = os.getenv("OPENAI_API_KEY") 
API_URL = "https://api.openai.com/v1/chat/completions" 

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
def ask_oracle_batch(frame_paths):
    if not API_KEY: return ["unknown"] * len(frame_paths)

    content = [{
        "type": "text",
        "text": "Classify the football action in these images sequentially. "
                "Return a JSON object with a key 'labels' containing a list of strings. "
                "Options: 'pass', 'shot', 'cross', 'duel', 'ball_loss', 'carry', 'none'. "
                "Example: {'labels': ['pass', 'duel', 'none']}"
    }]

    for p in frame_paths:
        b64 = encode_image(p)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
        })

    payload = {
        "model": "gpt-4o", 
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "response_format": { "type": "json_object" }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"   ⚠️ API Error {response.status_code}")
        raise Exception(f"API {response.status_code}")

    try:
        content_str = response.json()['choices'][0]['message']['content']
        parsed = json.loads(content_str)
        return parsed.get("labels", ["unknown"] * len(frame_paths))
    except:
        return ["unknown"] * len(frame_paths)

