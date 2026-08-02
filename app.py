from re import match
from quiz_explanations import QUIZ_EXPLANATIONS

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

import os
import traceback
import atexit

import joblib
import faiss
import mysql.connector
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

main_distilbert_path = "models/distilbert_emotion"

main_tokenizer = AutoTokenizer.from_pretrained(main_distilbert_path)

main_model = AutoModelForSequenceClassification.from_pretrained(
    main_distilbert_path
)

main_model.eval()

main_encoder = joblib.load("models/main_emotion_encoder.pkl")

sub_distilbert_path = "models/distilbert_sub_emotion"
sub_tokenizer = AutoTokenizer.from_pretrained(sub_distilbert_path)

sub_model = AutoModelForSequenceClassification.from_pretrained(
    sub_distilbert_path
)

sub_model.eval()

sub_encoder = joblib.load("models/sub_emotion_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_model.to(device)
sub_model.to(device)

print(f"Using device: {device}")


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

# Build FAISS index once for the full counselling dataset

all_embeddings = embedding_model.encode(
    counsel_df["Context"].tolist(),
    convert_to_numpy=True
).astype("float32")

full_index = faiss.IndexFlatL2(all_embeddings.shape[1])
full_index.add(all_embeddings)

print("✅ All models loaded successfully.")

quiz_df = pd.read_csv(
    "datasets/emotion_mcq_dataset.csv",
    encoding="utf-8"
)
# ==========================================
# Close Database Properly
# ==========================================

@atexit.register
def close_db():

    try:

        if db.is_connected():
            cursor.close()
            db.close()

    except Exception:
        pass

# ==========================================
# Update User Goal
# ==========================================

def update_goal(username, user_text):
    reconnect_db()

    text = user_text.lower()

    keywords = [
        "my goal is",
        "my aim is",
        "my dream is",
        "i want to become",
        "my career goal is"
    ]

    if any(keyword in text for keyword in keywords):

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
    

def generate_initial_response(
    main_emotion,
    sub_emotion,
    retrieved_responses
):

    intro = (
        f"I understand that you're experiencing "
        f"{sub_emotion} under the emotion "
        f"{main_emotion}.\n\n"
    )
    body = ""

    for i, response in enumerate(retrieved_responses, start=1):
        body += f"{i}. {response}\n\n"

    closing = (
        "These suggestions were selected "
        "from our counselling knowledge base."
    )
    return intro + body + closing


def retrieve_best_counselling(search_df, indices):
    responses = []
    used = set()

    for idx in indices[0]:

        if idx == -1:
            continue

        if idx >= len(search_df):
            continue
        response = search_df.iloc[idx]["Response"]

        if response not in used:
            responses.append(response)
            used.add(response)

        if len(responses) == 5:
            break

    return responses

def fallback_main_emotion(text):
    text = text.lower()

    # ---------------- SAD ----------------
    sad_patterns = [
        "i failed", "failed exam", "failed interview",
        "breakup", "heartbroken", "cry", "crying",
        "lonely", "depressed", "hopeless",
        "disappointed", "upset", "loss", "lost",
        "rejected", "missed opportunity","nobody loves me",
        "no one loves me","worthless","alone","left me","ignored"
    ]
    if any(p in text for p in sad_patterns):
        return "Sad"

    # ---------------- FEAR ----------------
    fear_patterns = [
        "exam tomorrow", "interview tomorrow",
        "afraid", "fear", "scared",
        "panic", "anxious", "nervous",
        "worried", "stress", "stressed",
        "tension", "terrified"
    ]
    if any(p in text for p in fear_patterns):
        return "Fear"

    # Generic exam/interview mention
    if any(word in text for word in ["exam", "interview", "deadline", "test"]):
        return "Fear"

    # ---------------- ANGRY ----------------
    angry_patterns = [
        "angry", "furious", "frustrated",
        "hate", "annoyed", "irritated",
        "mad", "betrayed", "false accusation",
        "accused", "insulted", "offended","stolen",
        "took my wallet","cheated","lied","blamed me"
    ]
    if any(p in text for p in angry_patterns):
        return "Angry"

    # ---------------- AFFECTION ----------------
    affection_patterns = [
        "love", "loved", "loving",
        "care", "caring", "hug",
        "kiss", "miss you", "adore",
        "girlfriend", "boyfriend",
        "family", "friendship", "romantic"
    ]
    if any(p in text for p in affection_patterns):
        return "Affection"

    # ---------------- RELIEF ----------------
    relief_patterns = [
        "finally", "thank god",
        "it's over", "its over",
        "escaped", "safe now",
        "recovered", "finished successfully",
        "problem solved", "relieved"
    ]
    if any(p in text for p in relief_patterns):
        return "Relief"

    # ---------------- EMBARRASSMENT ----------------
    embarrassment_patterns = [
        "embarrassed", "awkward",
        "ashamed", "humiliated",
        "everyone laughed", "blushed",
        "made fun of", "publicly embarrassed"
    ]
    if any(p in text for p in embarrassment_patterns):
        return "Embarrassment"

    # ---------------- CURIOSITY ----------------
    curiosity_patterns = [
        "curious", "wonder",
        "interested", "explore",
        "discover", "learn",
        "why", "how", "what if"
    ]
    if any(p in text for p in curiosity_patterns):
        return "Curiosity"

    # ---------------- HAPPY ----------------
    happy_patterns = [
        "i cracked", "i passed", "i got selected", "i got the job",
        "i got internship", "i got the internship", "promotion",
        "won", "victory", "achievement", "success",
        "celebrate", "party", "excited", "happy",
        "great news", "good news", "dream come true"
    ]
    if any(p in text for p in happy_patterns):
            return "Happy"

    # ---------------- DEFAULT ----------------
    return "Neutral"


def generate_wellness_card(main_emotion, sub_emotion):

    prompt = f"""
You are a mental wellness coach.

The detected emotion is:

Main Emotion: {main_emotion}
Sub Emotion: {sub_emotion}

Generate ONE personalized wellness card.
Return exactly in this format:

🌱 Focus:
...

💨 Exercise:
...

📝 Reflection:
...

🎯 Tiny Goal:
...

💬 Reminder:
...

Keep it under 120 words.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

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
        (
            name,
            age,
            gender,
            occupation,
            goal,
            recurring_emotion
            ) = profile
    else:
        (
            name,
            age,
            gender,
            occupation,
            goal,
            recurring_emotion
            ) = (
                "User",
                "",
                "",
                "",
                "",
                ""
            )

    # ==========================================
    # # Emotion Prediction
    # # ==========================================

    # ---------- Main Emotion (DistilBERT) ----------
    main_inputs = main_tokenizer(
        user_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
        )
    
    main_inputs = {k: v.to(device) for k, v in main_inputs.items()}
    with torch.no_grad():
        main_outputs = main_model(**main_inputs)
        probs = torch.softmax(main_outputs.logits, dim=1)

    main_pred = torch.argmax(probs, dim=1).item()
    main_confidence = probs[0][main_pred].item()
    main_emotion = main_encoder.inverse_transform([main_pred])[0]
    
    if main_confidence < 0.60:
        print(
            f"Low confidence ({main_confidence:.2f}) "
            f"→ Using fallback instead of {main_emotion}"
        )
        main_emotion = fallback_main_emotion(user_text)
    else:
        print(
            f"High confidence ({main_confidence:.2f}) "
            f"→ Using model prediction: {main_emotion}"
        )

    # ---------- Sub Emotion (DistilBERT) ----------
    sub_inputs = sub_tokenizer(
        user_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    sub_inputs = {k: v.to(device) for k, v in sub_inputs.items()}

    with torch.no_grad():
        sub_outputs = sub_model(**sub_inputs)
        sub_pred = torch.argmax(sub_outputs.logits, dim=1).item()
    sub_emotion = sub_encoder.inverse_transform([sub_pred])[0]

    del main_inputs, main_outputs
    del sub_inputs, sub_outputs
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # RAG Retrieval

    # Use emotion filter only if prediction is confident
    if main_confidence >= 0.60:
        filtered_df = counsel_df[
            counsel_df["Main_Emotion"].str.lower()
            == main_emotion.lower()
            ]
        if len(filtered_df) > 5:
            search_df = filtered_df.reset_index(drop=True)
        else:
            search_df = counsel_df.reset_index(drop=True)
    else:
        #Low confidence → search whole counselling dataset
        search_df = counsel_df.reset_index(drop=True)

    query_embedding = embedding_model.encode(
        [user_text],
        convert_to_numpy=True
        ).astype("float32")
    
    temp_embeddings = embedding_model.encode(
        search_df["Context"].tolist(),
        convert_to_numpy=True
        ).astype("float32")
    
    temp_index = faiss.IndexFlatL2(temp_embeddings.shape[1])
    temp_index.add(temp_embeddings)

    distances, indices = temp_index.search(query_embedding, 5)

    best_responses = retrieve_best_counselling(search_df,indices)
    
    initial_response = generate_initial_response(
        main_emotion,
        sub_emotion,
        best_responses
        )
    rag_context = "\n\n".join(best_responses)
    print(
        f"Main: {main_emotion} ({main_confidence:.2f}) | Sub: {sub_emotion}")

    # Conversation Memory
    conversation = ""
    for msg in history or []:

        conversation += (
            f'{msg["role"]}: {msg["content"]}\n'
        )

    prompt = f"""
You are a Psychological Remedies AI Assistant.

Your primary knowledge comes from the counselling response generated by our counselling engine.

Do NOT ignore it.
Do NOT generate completely different advice.
Instead, improve it naturally.

User Profile:
Name: {name}
Age: {age}
Gender: {gender}
Occupation: {occupation}

Goal:
{goal}

Previous Recurring Emotion:
{recurring_emotion}

Conversation History:
{conversation}

Detected Main Emotion:
{main_emotion}

Detected Sub Emotion:
{sub_emotion}

Initial Counselling Response:
{initial_response}

Retrieved Counselling Context:
{rag_context}

Current User Message:
{user_text}

Instructions:

1. First understand the user's emotion.
2. Use the Initial Counselling Response as the primary answer.
3. Use the Retrieved Counselling Context only to strengthen or enrich the answer.
4. Personalize the response using the user's profile whenever appropriate.
5. Speak naturally and empathetically.
6. Never contradict the Initial Counselling Response.
7. Never invent personal facts.
8. If the user has a stored goal, relate the counselling to that goal whenever appropriate.
9. If the recurring emotion is the same as the current emotion, gently acknowledge that this feeling has appeared before.
10. Encourage gradual progress instead of giving generic advice.
11. Keep the response between 100 and 150 words.
12. End with one small practical action the user can take today.
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
        wellness_card = generate_wellness_card(
            main_emotion,
            sub_emotion
        )

        return main_emotion, sub_emotion, reply, wellness_card 

    except Exception:
        traceback.print_exc()

        return (
            main_emotion,
            sub_emotion,
            "Sorry, I'm temporarily unavailable. Please try again in a moment.",
            ""
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
            ai_reply,
            created_at
        FROM chat_history
        WHERE username=%s
        ORDER BY id ASC
    """, (username,))

    rows = cursor.fetchall()

    messages = []

    for user_msg, ai_msg, created_at in rows:

        messages.append({
            "role": "user",
            "content": user_msg,
            "time": str(created_at)
        })

        messages.append({
            "role": "assistant",
            "content": ai_msg,
            "time": str(created_at)
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

@app.route("/quiz", methods=["GET"])
def get_quiz():

    reconnect_db()

    username = request.args.get("username")

    if not username:
        return jsonify({"error": "Username required"}), 400

    # Get attempted question IDs
    cursor.execute("""
        SELECT question_id
        FROM quiz_history
        WHERE username=%s
    """, (username,))

    attempted = [row[0] for row in cursor.fetchall()]

    # Quiz completed
    if len(attempted) >= 10:
        return jsonify({
            "completed": True,
            "message": "Quiz Completed"
        })

    # Remove attempted questions
    remaining = quiz_df[~quiz_df["id"].isin(attempted)]

    # Safety check
    if remaining.empty:
        return jsonify({
            "completed": True,
            "message": "Quiz Completed"
        })

    # Pick one remaining question
    question = remaining.sample(1).iloc[0]

    return jsonify({
        "completed": False,
        "id": int(question["id"]),
        "scenario": question["scenario"],
        "option1": question["option1"],
        "option2": question["option2"],
        "option3": question["option3"],
        "option4": question["option4"],
        "difficulty": question["difficulty"]
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

@app.route("/quiz_answer", methods=["POST"])
def quiz_answer():

    reconnect_db()

    data = request.json

    qid = data["id"]
    selected = data["selected"]
    username = data["username"]

    match = quiz_df[quiz_df["id"] == qid]
    if match.empty:
        return jsonify({"error": "Question not found"}), 404
    row = match.iloc[0]

    correct = row["correct_answer"]
    explanation = QUIZ_EXPLANATIONS.get(
        int(qid),
        "No explanation available."
        )
    # Duplicate attempt check

    cursor.execute("""
    SELECT COUNT(*)
    FROM quiz_history
    WHERE username=%s AND question_id=%s
    """, (username, qid))

    already = int(cursor.fetchone()[0] or 0)

    if already>0:
        return jsonify({"error": "Question already attempted"}), 400

    if selected not in [
        row["option1"],
        row["option2"],
        row["option3"],
        row["option4"]
    ]:
        return jsonify({"error": "Invalid option"}), 400

    cursor.execute("""
    INSERT INTO quiz_history
    (username, question_id, selected_answer, correct_answer, is_correct)
    VALUES (%s, %s, %s, %s, %s)
    """,
    (
        username,
        qid,
        selected,
        correct,
        selected == correct
    ))

    db.commit()

    return jsonify({
        "correct": selected == correct,
        "correct_answer": correct,
        "explanation": explanation,
        "score": 1 if selected == correct else 0
    })

@app.route("/reset_quiz", methods=["POST"])
def reset_quiz():

    reconnect_db()

    data = request.json

    cursor.execute("""
    DELETE FROM quiz_history
    WHERE username=%s
    """, (data["username"],))

    db.commit()

    return jsonify({"status": "success"})

@app.route("/quiz_score", methods=["GET"])
def quiz_score():

    reconnect_db()

    username = request.args.get("username")

    cursor.execute("""
    SELECT
    COUNT(*),
    SUM(is_correct)
    FROM quiz_history
    WHERE username=%s
    """, (username,))

    attempted, correct = cursor.fetchone()

    attempted = int(attempted or 0)
    correct = int(correct or 0)

    accuracy = round((correct / attempted) * 100, 2) if attempted else 0

    return jsonify({
        "attempted": attempted,
        "correct": correct,
        "accuracy": accuracy
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

    main_emotion, sub_emotion, reply, wellness_card = get_ai_response(
        username,
        user_message,
        history
    )
    full_reply = reply
    if wellness_card:
        full_reply += (
            "\n\n---\n\n"
            "## 🌱 Personalized Wellness Card\n\n"
            + wellness_card
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
        full_reply
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
        "reply": reply,
        "wellness_card": wellness_card
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