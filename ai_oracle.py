import base64
import os
from openai import OpenAI

# You need to set this in your .env file
# export OPENAI_API_KEY="sk-..."
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ask_chatgpt_vision(image_path, prompt):
    """
    Sends an image to GPT-4o to get a human-level label.
    """
    if not client:
        print("❌ SKIPPING ORACLE: No OpenAI API Key found.")
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
            max_tokens=50,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Oracle Error: {e}")
        return None
