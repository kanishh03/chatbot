# app.py
from flask import Flask, request, jsonify, render_template, send_file
import pickle
import re
from transformers import pipeline
import datetime
from gtts import gTTS
import tempfile
import os
import io

app = Flask(__name__)

# =========================================================
# LOAD MODELS
# =========================================================
try:
    sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    sentiment_model = None
    vectorizer = None

# Initialize Emotion AI Pipeline
try:
    emotion_ai = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=3
    )
except Exception as e:
    print(f"Error loading emotion pipeline: {e}")
    emotion_ai = None

# =========================================================
# HELPER FUNCTIONS (Ported from Streamlit app)
# =========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_sentiment(text):
    if not sentiment_model or not vectorizer:
        return "Neutral", 0.0
        
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = sentiment_model.predict(vec)[0]
    prob = sentiment_model.predict_proba(vec).max() * 100
    label = "Positive" if pred == 1 else "Negative"
    return ("Neutral", round(prob, 2)) if prob < 60 else (label, round(prob, 2))

EMERGENCY_WORDS = [
    "suicide", "kill myself", "end my life",
    "i want to die", "self harm"
]

def detect_emergency(text):
    return any(w in text.lower() for w in EMERGENCY_WORDS)

def nurse_reply(sentiment, negative_count=0):
    if sentiment == "Emergency":
        return (
            "🚨 I’m really concerned about your safety.<br><br>"
            "📞 AASRA (India): 91-9820466726<br>"
            "📞 Emergency: 112<br><br>"
            "You are not alone."
        )
    # Simplified logic for stateless API. 
    # For complex state tracking (negative count), we'd need a DB or session.
    if sentiment == "Negative":
        return "💭 That sounds really difficult. Want to share more?"
    if sentiment == "Positive":
        return "😊 I’m glad to hear that. What helped today?"
    return "🙂 I’m listening."

# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # 1. Emergency Detection
    if detect_emergency(user_input):
        sentiment, confidence = "Emergency", 100
    else:
        sentiment, confidence = predict_sentiment(user_input)

    # 2. Emotion Analysis
    emotion_result = []
    if emotion_ai:
        emotions = emotion_ai(user_input)[0]
        # Format: [{'label': 'joy', 'score': 0.9}, ...]
        emotion_result = [{"label": e['label'], "score": round(e['score'], 2)} for e in emotions]
    
    # 3. Generate Reply
    reply = nurse_reply(sentiment) # Note: Negative count logic omitted for stateless simplicity

    response = {
        "reply": reply,
        "sentiment": sentiment,
        "confidence": confidence,
        "emotions": emotion_result,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return jsonify(response)

# Note: PDF generation requires passing full history from frontend or storing it in DB.
# Skipping for this stateless simplified migration.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
