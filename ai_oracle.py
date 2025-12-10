import base64
import os
import json
import cv2
from openai import OpenAI

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ask_oracle(image_path, prompt):
    """
    Sends a frame to ChatGPT-4o Vision for expert analysis.
    """
    if not OPENAI_API_KEY:
        print("❌ No API Key found.")
        return None

    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Oracle Error: {e}")
        return None

# --- INTEGRATION TEST ---
if __name__ == "__main__":
    # Test with a dummy frame if one exists
    print("🔮 Oracle System Initialized.")

