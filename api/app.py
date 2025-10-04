import re
from flask import Flask, request, jsonify, render_template # type: ignore
from dotenv import load_dotenv # type: ignore
from huggingface_hub import InferenceClient # type: ignore

load_dotenv()

app = Flask(__name__)

# Configure the Hugging Face client
client = InferenceClient()
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

@app.route("/")
def index():
    return render_template("index.html")

# In app.py
# In app.py
# In app.py
def generate_subject_lines(email_body):
    prompt = f"""
    [INST]
    Generate 3 engaging and distinct subject lines for the following email body.
    Provide the output as a numbered list. Do not use emojis or any markdown formatting.
    
    Email Body:
    {email_body}
    [/INST]
    """

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_NAME,
        max_tokens=100,
    )

    content = response.choices[0].message.content.strip()
    
    suggestions = [line.strip().split('. ', 1)[-1] for line in content.split('\n') if line.strip()]
    suggestions = suggestions[:3]

    # This filter now removes BOTH emojis AND asterisks.
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    cleaned_suggestions = [emoji_pattern.sub(r'', s).replace('*', '').strip() for s in suggestions]
    
    return cleaned_suggestions

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    if not data or 'email_body' not in data:
        return jsonify({"error": "Missing 'email_body' in request"}), 400
        
    email_body = data.get("email_body", "")
    try:
        suggestions = generate_subject_lines(email_body)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        # We can keep the error printer just in case
        print("----------- !!! CAUGHT AN ERROR !!! -----------")
        print(f"The specific Hugging Face error is: {e}")
        print("---------------------------------------------")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)