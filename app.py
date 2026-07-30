from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import traceback
import atexit

import joblib
import faiss
import mysql.connector

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)
CORS(app)

# ==========================================
# Groq Client
# ==========================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ==========================================
# MySQL Connection
# ==========================================

db = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)

cursor = db.cursor(buffered=True)

# ==========================================
# Helper Function
# ==========================================

def reconnect_db():
    global db, cursor

    if not db.is_connected():

        db = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )

        cursor = db.cursor(buffered=True)

# ==========================================
# Load Emotion Models
# ==========================================

tfidf = joblib.load("models/tfidf_vectorizer.pkl")

emotion_model = joblib.load(
    "models/main_emotion_model_tfidf.pkl"
)

main_encoder = joblib.load(
    "models/main_emotion_encoder.pkl"
)

sub_model = joblib.load(
    "models/sub_emotion_model_tfidf.pkl"
)

sub_encoder = joblib.load(
    "models/sub_emotion_encoder.pkl"
)

# ==========================================
# Load RAG
# ==========================================

index = faiss.read_index(
    "models/counseling_faiss.index"
)

counsel_df = joblib.load(
    "models/counseling_dataset.pkl"
)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L12-v2"
)

print("✅ All models loaded successfully.")

# ==========================================
# Close Database Properly
# ==========================================

@atexit.register
def close_db():

    try:

        if db.is_connected():
            cursor.close()
            db.close()

    except:
        pass

# ==========================================
# Update User Goal
# ==========================================

def update_goal(username, user_text):

    reconnect_db()

    keywords = [
        "goal",
        "dream",
        "want",
        "aim",
        "career",
        "placement",
        "job",
        "exam",
        "study",
        "future"
    ]

    text = user_text.lower()

    for word in keywords:

        if word in text:

            cursor.execute("""
                UPDATE user_profile
                SET goal=%s
                WHERE username=%s
            """,
            (
                user_text,
                username
            ))

            db.commit()
            break


# ==========================================
# AI Response Function
# ==========================================

def get_ai_response(username, user_text, history):

    reconnect_db()

    cursor.execute("""
        SELECT
            name,
            age,
            gender,
            occupation,
            goal,
            recurring_emotion
        FROM user_profile
        WHERE username=%s
    """, (username,))

    profile = cursor.fetchone()

    if profile:

        name = profile[0]
        age = profile[1]
        gender = profile[2]
        occupation = profile[3]
        goal = profile[4]
        recurring_emotion = profile[5]

    else:

        name = "User"
        age = ""
        gender = ""
        occupation = ""
        goal = ""
        recurring_emotion = ""

    # Emotion Prediction

    x = tfidf.transform([user_text])

    main_pred = emotion_model.predict(x)[0]
    main_emotion = main_encoder.inverse_transform([main_pred])[0]

    sub_pred = sub_model.predict(x)[0]
    sub_emotion = sub_encoder.inverse_transform([sub_pred])[0]

    # RAG Retrieval

    query_embedding = embedding_model.encode(
        [user_text],
        convert_to_numpy=True
    )

    distances, indices = index.search(query_embedding, 3)

    contexts = []

    for idx in indices[0]:
        contexts.append(counsel_df.iloc[idx]["Response"])

    rag_context = "\n\n".join(contexts)

    # Conversation Memory

    conversation = ""

    for msg in history or []:

        conversation += (
            f'{msg["role"]}: {msg["content"]}\n'
        )

    prompt = f"""
You are a compassionate Psychological Remedies Chatbot.

User Profile

Name: {name}
Age: {age}
Gender: {gender}
Occupation: {occupation}
Goal: {goal}
Recurring Emotion: {recurring_emotion}

Conversation History:
{conversation}

Detected Main Emotion:
{main_emotion}

Detected Sub Emotion:
{sub_emotion}

Relevant Counseling Knowledge:
{rag_context}

Current User Message:
{user_text}

Instructions:

- Reply empathetically.
- Personalize using the user's profile.
- If a goal exists, encourage progress toward it.
- If the recurring emotion matches today's emotion, gently acknowledge the pattern.
- Never invent personal information.
- Give practical coping suggestions.
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {
                    "role": "system",
                    "content": "You are a helpful psychological counsellor."
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ]

        )

        reply = response.choices[0].message.content

        return main_emotion, sub_emotion, reply

    except Exception:

        traceback.print_exc()

        return (
            main_emotion,
            sub_emotion,
            "Sorry, I'm temporarily unavailable. Please try again in a moment."
        )

# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():
    return "Psychological Remedies Chatbot Backend Running"


# ==========================================
# CHAT HISTORY
# ==========================================

@app.route("/history", methods=["GET"])
def history():

    reconnect_db()

    username = request.args.get("username")

    cursor.execute("""
        SELECT
            user_message,
            ai_reply
        FROM chat_history
        WHERE username=%s
        ORDER BY id ASC
    """, (username,))

    rows = cursor.fetchall()

    messages = []

    for user_msg, ai_msg in rows:

        messages.append({
            "role": "user",
            "content": user_msg
        })

        messages.append({
            "role": "assistant",
            "content": ai_msg
        })

    return jsonify(messages)


# ==========================================
# PROFILE
# ==========================================

@app.route("/profile", methods=["GET"])
def profile():

    reconnect_db()

    username = request.args.get("username")

    cursor.execute("""
        SELECT
            name,
            age,
            gender,
            occupation,
            goal,
            recurring_emotion
        FROM user_profile
        WHERE username=%s
    """, (username,))

    row = cursor.fetchone()

    if row is None:

        return jsonify({

            "name": "",
            "age": 0,
            "gender": "",
            "occupation": "",
            "goal": "",
            "recurring_emotion": ""

        })

    return jsonify({

        "name": row[0],
        "age": row[1],
        "gender": row[2],
        "occupation": row[3],
        "goal": row[4],
        "recurring_emotion": row[5]

    })


# ==========================================
# UPDATE PROFILE
# ==========================================

@app.route("/update_profile", methods=["POST"])
def update_profile():

    reconnect_db()

    data = request.json

    cursor.execute("""
        UPDATE user_profile
        SET
            name=%s,
            age=%s,
            gender=%s,
            occupation=%s,
            goal=%s
        WHERE username=%s
    """,
    (
        data["name"],
        data["age"],
        data["gender"],
        data["occupation"],
        data["goal"],
        data["username"]
    ))

    db.commit()

    return jsonify({

        "status": "success"

    })

# ==========================================
# CHAT
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    reconnect_db()

    data = request.get_json()

    if data is None:
        return jsonify({"reply": "Invalid Request"}), 400

    username = data.get("username", "demo_user")
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if user_message == "":
        return jsonify({"reply": "Please enter a message."})

    main_emotion, sub_emotion, reply = get_ai_response(
        username,
        user_message,
        history
    )

    cursor.execute("""
        INSERT INTO chat_history
        (
            username,
            user_message,
            main_emotion,
            sub_emotion,
            ai_reply
        )
        VALUES
        (%s,%s,%s,%s,%s)
    """,
    (
        username,
        user_message,
        main_emotion,
        sub_emotion,
        reply
    ))

    db.commit()

    cursor.execute("""
        UPDATE user_profile
        SET recurring_emotion=%s
        WHERE username=%s
    """,
    (
        main_emotion,
        username
    ))

    db.commit()

    update_goal(username, user_message)

    return jsonify({
        "main_emotion": main_emotion,
        "sub_emotion": sub_emotion,
        "reply": reply
    })


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )