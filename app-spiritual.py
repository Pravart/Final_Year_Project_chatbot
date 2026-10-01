from pydoc import text
from re import match
from app import is_mental_health_conversation
from quiz_explanations import QUIZ_EXPLANATIONS
import numpy as np

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
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

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

main_distilbert_path = "models/distilbert_emotion_onnx"

main_tokenizer = AutoTokenizer.from_pretrained(main_distilbert_path)

main_model = ORTModelForSequenceClassification.from_pretrained(
    "models/distilbert_emotion_onnx",
    file_name="model_int8.onnx"
)

main_encoder = joblib.load("models/main_emotion_encoder.pkl")

sub_distilbert_path = "models/distilbert_sub_emotion_onnx"
sub_tokenizer = AutoTokenizer.from_pretrained(sub_distilbert_path)

sub_model = ORTModelForSequenceClassification.from_pretrained(
    "models/distilbert_sub_emotion_onnx",
    file_name="model_int8.onnx"
)

mental_classifier = joblib.load("models/mental_classifier.pkl")
mental_vectorizer = joblib.load("models/mental_vectorizer.pkl")

sub_encoder = joblib.load("models/sub_emotion_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_model.to(device)
sub_model.to(device)

print(f"Using device: {device}")

# ==========================================
# Load RAG
# ==========================================

counsel_df = joblib.load("models/counseling_dataset.pkl")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Load precomputed counselling embeddings
all_embeddings = np.load("models/counselling_embeddings.npy").astype("float32")

# Build FAISS index from precomputed embeddings
COUNSELLING_INDEX = faiss.IndexFlatL2(all_embeddings.shape[1])

COUNSELLING_INDEX.add(all_embeddings)

print("✅ RAG model, embeddings and FAISS index loaded successfully.")
# ==========================================
# Load Spiritual & Mythological Wellness Dataset
# ==========================================
SPIRITUAL_DATASET_PATH = "datasets/spiritual_wellness_dataset.csv"

spiritual_df = pd.read_csv(
    SPIRITUAL_DATASET_PATH,
    encoding="cp1252"
)

print(f"✅ Spiritual wellness dataset loaded: "
      f"{len(spiritual_df)} rows"
)

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
        "no one loves me","worthless","alone","left me","ignored","low score",
        "low scores","low mark","low marks","poor marks","poor score","poor scores",
        "scored low","got low","did badly","didn't get good marks",
        "couldn't get good marks","got average marks","got low marks",
        "I am feeling low","I feel low","I feel low today","I'm feeling down",
        "I feel emotionally low","I'm feeling really low lately",
        "I have been feeling low for days",
        "I don't feel like doing anything today",
        "My mood is very low.","uplift my mood","improve my mood","feeling down","down lately",
        "not feeling good","feeling bad","cheer me up","how to feel better","help my mood",
        "motivate me","need motivation","feeling empty","feeling miserable",
    ]
    if any(p in text for p in sad_patterns):
        return "Sad"

    # ---------------- FEAR ----------------
    fear_patterns = [
        "exam tomorrow", "interview tomorrow",
        "afraid", "fear", "scared",
        "panic", "anxious", "nervous",
        "worried", "stress", "stressed",
        "tension", "terrified","tomorrow","next week",
        "upcoming","going to","have an exam",
        "have a test","interview","i want good marks","i hope i pass",
        "i want to become topper","i have exam tomorrow","overthinking",
        "overthink","future","barely sleep","can't sleep","cannot sleep",
        "heart races","racing heart","pretending","everything is okay"
    ]
    if any(p in text for p in fear_patterns):
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
        "great news", "good news", "dream come true","i got good marks",
        "i scored good marks","i scored high marks","i became topper",
        "i got distinction","i received distinction","i passed"
    ]
    if any(p in text for p in happy_patterns):
            return "Happy"

    # ---------------- DEFAULT ----------------
    return "Neutral"

def fallback_sub_emotion(text, main_emotion):
    text = text.lower()

    # ================= FEAR =================
    if main_emotion == "Fear":

        nervousness = [
            "exam","test","quiz","interview","viva","presentation",
            "tomorrow","deadline","result","results","performance",
            "job","placement","selection","competition","competition",
            "speech","meeting","stage","public speaking","first day",
            "worried","nervous","tense","stress","stressed","panic",
            "anxious","anxiety","can't sleep","cannot sleep"
        ]

        fear = [
            "afraid","fear","scared","terrified","frightened",
            "horror","danger","unsafe","threat","accident",
            "kidnap","robbery","earthquake","death","die",
            "hospital","injury","attack","violence","crime"
        ]

        if any(k in text for k in nervousness):
            return "nervousness"

        if any(k in text for k in fear):
            return "fear"

        return "fear"

    # ================= SAD =================
    elif main_emotion == "Sad":

        grief = [
            "alone","lonely","nobody","no one","hopeless",
            "worthless","empty","abandoned","left me","miss you",
            "death","passed away","funeral","lost my father",
            "lost my mother","lost my friend","broken family"
        ]

        disappointment = [
            "failed","failure","didn't get","did not get",
            "not selected","rejected","low marks","bad marks",
            "disappointed","missed opportunity","couldn't achieve",
            "could not achieve","unsuccessful"
        ]

        remorse = [
            "my mistake","my fault","i regret","i am sorry",
            "guilty","regret","shouldn't have","should not have",
            "forgive me","i apologize"
        ]

        sadness = [
            "sad","cry","crying","depressed","upset",
            "hurt","pain","heartbroken","broken","feeling low"
        ]

        if any(k in text for k in grief):
            return "grief"

        if any(k in text for k in remorse):
            return "remorse"

        if any(k in text for k in disappointment):
            return "disappointment"

        if any(k in text for k in sadness):
            return "sadness"

        return "sadness"

    # ================= HAPPY =================
    elif main_emotion == "Happy":

        gratitude = [
            "thank","thanks","thank you","grateful",
            "appreciate","blessed"
        ]

        admiration = [
            "inspired","admire","respect","idol","role model"
        ]

        excitement = [
            "excited","can't wait","cannot wait","looking forward",
            "thrilled","amazing","awesome"
        ]

        joy = [
            "happy","passed","cracked","selected","won",
            "success","promotion","celebrate","party",
            "achievement","good news"
        ]

        if any(k in text for k in gratitude):
            return "gratitude"

        if any(k in text for k in admiration):
            return "admiration"

        if any(k in text for k in excitement):
            return "excitement"

        if any(k in text for k in joy):
            return "joy"

        return "joy"

    # ================= ANGRY =================
    elif main_emotion == "Angry":

        annoyance = [
            "annoy","annoyed","irritated","disturbed",
            "bothered","frustrated","fed up"
        ]

        anger = [
            "angry","mad","furious","rage","hate",
            "shouting","yelling","temper"
        ]

        disgust = [
            "disgust","gross","dirty","filthy","nasty"
        ]

        disapproval = [
            "wrong","unfair","cheated","betrayed",
            "lied","lie","fake","corrupt","stolen","my wallet","took my wallet"
        ]

        if any(k in text for k in anger):
            return "anger"

        if any(k in text for k in annoyance):
            return "annoyance"

        if any(k in text for k in disgust):
            return "disgust"

        if any(k in text for k in disapproval):
            return "disapproval"

        return "anger"

    # ================= AFFECTION =================
    elif main_emotion == "Affection":

        love = [
            "love","beloved","girlfriend","boyfriend",
            "wife","husband","romantic","kiss","hug",
            "care","caring","miss you","adore"
        ]

        caring = [
            "family","friend","support","help","protect",
            "kind","kindness","comfort","take care"
        ]

        if any(k in text for k in love):
            return "love"

        if any(k in text for k in caring):
            return "caring"

        return "love"

    # ================= RELIEF =================
    elif main_emotion == "Relief":

        if any(k in text for k in [
            "finally","relief","relieved","safe","escaped","problem solved","finished",
            "it's over","its over","completed","recovered","thank god","thank goodness"
        ]):
            return "relief"

        return "relief"

    # ================= CURIOSITY =================
    elif main_emotion == "Curiosity":

        if any(k in text for k in [
            "curious","wonder","interested","discover","explore","learn",
            "why","how","what if","explain","tell me","can you",
            "question","research"
        ]):
            return "curiosity"

        return "curiosity"

    # ================= EMBARRASSMENT =================
    elif main_emotion == "Embarrassment":

        if any(k in text for k in [
            "embarrassed","awkward","ashamed",
            "humiliated","blushed",
            "everyone laughed","made fun",
            "publicly embarrassed"
        ]):
            return "embarrassment"
        return "embarrassment"
    # ================= PHYSICAL WELLBEING =================
    elif main_emotion == "Neutral":
        physical_symptoms = [
            "headache","head ache","migraine","insomnia","can't sleep","cannot sleep",
            "sleep problem","tired","exhausted","fatigue","low energy","body pain",
            "stomach pain"
        ]
        if any(k in text for k in physical_symptoms):
            return "Physical Wellbeing"
        return "neutral"
    return "neutral"

def retrieve_spiritual_context(main_emotion, sub_emotion, user_text):
    try:
        # First filter by detected main emotion
        filtered_df = spiritual_df[
            spiritual_df["main_emotion"].str.lower()
            == str(main_emotion).lower()
        ].copy()

        if filtered_df.empty:
            return None

        # Build searchable text from situation + sub-emotion
        search_texts = (
            filtered_df["sub_emotion"].fillna("").astype(str)
            + " "
            + filtered_df["situation"].fillna("").astype(str)
        ).tolist()

        # Use the SentenceTransformer already loaded for counselling RAG
        candidate_embeddings = embedding_model.encode(
            search_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_text = f"{sub_emotion} {user_text}"

        query_embedding = embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # Cosine similarity
        similarities = np.dot(
            candidate_embeddings,
            query_embedding
        )

        best_index = int(np.argmax(similarities))
        best_row = filtered_df.iloc[best_index]

        return {
            "source": str(best_row["source"]),
            "spiritual_reference": str(
                best_row["spiritual_reference"]
            ),
            "spiritual_reflection": str(
                best_row["spiritual_reflection"]
            ),
        }

    except Exception as e:
        print("Spiritual retrieval error:", e)
        return None

    
def generate_wellness_card(
    main_emotion,
    sub_emotion,
    main_confidence,
    sub_confidence,
    user_text,
    spiritual_preference="No",
    spiritual_context=None
):

    if main_confidence >= 0.45:

        emotion_instruction = f"""
The emotion model has reasonably high confidence.

Main Emotion: {main_emotion}
Main Emotion Confidence: {main_confidence:.2f}

Sub Emotion: {sub_emotion}
Sub Emotion Confidence: {sub_confidence:.2f}

Use these detected emotions as the primary guidance,
but still consider the user's actual message before
generating the wellness card.
"""

    else:

        emotion_instruction = f"""
The emotion model has LOW confidence.

Model Main Emotion: {main_emotion}
Model Main Emotion Confidence: {main_confidence:.2f}

Model Sub Emotion: {sub_emotion}
Model Sub Emotion Confidence: {sub_confidence:.2f}

Do NOT blindly trust the predicted emotions.

Instead, analyze the user's actual message carefully.
Groq should independently determine what emotional,
mental-wellness, or practical support is most appropriate.

The model prediction is only a weak signal in this case.
The user's actual words and context should take priority.
"""
    # ------------------------------------------
    # Spiritual & Mythological section
    # ------------------------------------------

    spiritual_instruction = ""
    spiritual_format = ""

    if (
        str(spiritual_preference).lower() == "yes"
        and spiritual_context
    ):

        spiritual_instruction = f"""
The user has opted in to Spiritual & Mythological
Wellness Guidance.

Relevant retrieved spiritual context:

Source:
{spiritual_context["source"]}

Spiritual/Mythological Reference:
{spiritual_context["spiritual_reference"]}

Reflection:
{spiritual_context["spiritual_reflection"]}

Use this retrieved context to create a short,
relevant Spiritual & Mythological reflection.

Do not invent verses, quotations, chapter numbers,
religious events, or scriptural claims.

Do not copy the retrieved reflection mechanically.
Adapt its principle naturally to the user's situation.
"""

        spiritual_format = """
🕉️ Spiritual & Mythological:
...
"""


    prompt = f"""
You are a mental wellness coach inside an AI-powered
Psychological Remedies Chatbot.

User's message:
{user_text}

{emotion_instruction}

{spiritual_instruction}
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

{spiritual_format}

Rules:

1. Make the card relevant to the user's actual message.
2. Understand the user's situation before choosing the advice.
3. If model confidence is high, use the detected emotions
   as the primary guidance.
4. If model confidence is low, prioritize your own
   interpretation of the user's actual message.
5. Do not mention confidence scores to the user.
6. Do not state that the detected emotion is certain.
7. Do not invent personal facts.
8. Keep the advice practical, supportive, and concise.
9. If Spiritual & Mythological guidance is enabled,
   use only the retrieved spiritual context as the basis
   for that section.
10. Do not fabricate scripture, quotations, verses,
    chapter numbers, or mythological events.
11. If Spiritual & Mythological guidance is not enabled,
    do NOT include that section.
12. Keep the complete card concise.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful mental wellness coach."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception:
        traceback.print_exc()
        return ""

def is_mental_health_query(text):
    X = mental_vectorizer.transform([text])
    pred = mental_classifier.predict(X)[0]
    return pred == 1

import re

def secondary_mental_check(text):
    """
    Conservative fallback for obvious emotional, psychological,
    or wellbeing-related messages that the trained mental/non-mental
    classifier may miss.

    This is a fallback, NOT the primary classifier.
    """

    text = str(text).lower().strip()

    wellbeing_symptoms = [
        "headache","headaches","migraine","body pain","fatigue","exhausted","exhaustion",
        "tired","can't sleep","cannot sleep","insomnia","sleep problem","sleep problems",
        "loss of appetite","appetite problem","restless","restlessness","physical symptoms",
        "feeling unwell","stomach pain","nausea","dizziness","lightheaded","chest pain"
    ]

    emotion_words = [
        # Happy / positive
        "happy","happiness","joy","joyful","excited","excitement","relieved","relief",

        # Sad
        "sad","sadness","unhappy","cry","crying","upset",

        # Fear / anxiety
        "fear","fearful","scared","afraid","anxious","anxiety","nervous","nervousness","worried",
        "worry","panic","panicked",

        # Stress / tension
        "stress","stressed","tension","tense","overwhelmed","pressure",

        # Anger
        "angry","anger","furious","irritated","irritation","frustrated","frustration",

        # Relaxation / calmness
        "relaxed","relax","calm","peaceful",

        # Depression / loneliness
        "depressed","depression","lonely","loneliness","hopeless","helpless","worthless","empty",

        # Thinking / confusion
        "overthinking","overthink","confused","confusion",

        # Social/self-conscious emotions
        "jealous","jealousy","guilty","guilt","embarrassed","embarrassment","ashamed",

        # Motivation / confidence
        "motivation","motivated","unmotivated","confidence","confident","insecure","insecurity",

        # General emotional terms
        "mood","mood swing","mood swings","emotion","emotions","emotional","feeling",
        "feelings","mental","psychological",
        # Energy / sleep
        "tired","exhausted","sleepy"
    ]

    # These are useful contextual terms, but they should NOT
    # independently make a message a mental-health query.
    life_events = [
        "exam","exams","college","school","study","studying","studies","marks","result",
        "failed","failure","pass","job","work","office","career","placement",
        "placements","family","friend","friends","relationship","breakup","parents",
        "salary","interview","interviews","competition","deadline"
    ]

    # -----------------------------------------
    # Whole-word / whole-phrase matching
    # -----------------------------------------

    def contains_term(term):
        pattern = r"\b" + re.escape(term) + r"\b"
        return re.search(pattern, text, re.IGNORECASE) is not None

    has_emotion = any(
        contains_term(word)
        for word in emotion_words
    )

    has_symptom = any(
        contains_term(word)
        for word in wellbeing_symptoms
    )

    # Kept for future contextual use.
    # A life event alone does NOT make a query mental-health related.
    has_event = any(
        contains_term(word)
        for word in life_events
    )

    # IMPORTANT:
    # Do not return has_event by itself.
    #
    # Example:
    # "When is my exam?" -> should not automatically be mental.
    # "I am nervous about my exam." -> has_emotion = True.
    #
    # Your trained LR model is still checked separately.
    return has_emotion or has_symptom


def is_mental_health_conversation(user_text, history):
    """
    Final scope check.

    Layer 1:
        Trained mental/non-mental Logistic Regression model.

    Layer 2:
        Conservative keyword fallback.

    Layer 3:
        Conversation-aware handling for genuine follow-ups.
    """

    user_text = str(user_text).strip()

    # -----------------------------------------
    # 1. PRIMARY CHECK:
    #    Trained Logistic Regression classifier
    # -----------------------------------------

    if is_mental_health_query(user_text):
        return True

    # -----------------------------------------
    # 2. SECONDARY CHECK:
    #    Explicit mental/emotional/wellbeing language missed by LR
    # -----------------------------------------

    if secondary_mental_check(user_text):
        return True

    # -----------------------------------------
    # 3. Check whether current message appears to be a conversational follow-up
    # -----------------------------------------

    text = user_text.lower()

    follow_up_phrases = [
        "what should i do","what do i do","what to do","what can i do","how do i",
        "how can i","how should i","why is this","why does this","why am i","still feeling",
        "feel the same","feeling the same","again","right now","now what","help me",
        "this feeling","that feeling","same feeling","same problem","tell me more",
        "what about this","how to handle this","how to deal with this",
        "how to control this","it happened again","it's happening again",
        "it is happening again","what about now"
    ]

    looks_like_follow_up = any(
        phrase in text
        for phrase in follow_up_phrases
    )

    # If the current message is neither independently mental-health related nor a likely follow-up,
    # reject it as outside scope.
    if not looks_like_follow_up:
        return False

    # -----------------------------------------
    # 4. Examine recent USER messages
    # -----------------------------------------

    recent_user_messages = []

    if not isinstance(history, list):
        history = []

    for message in reversed(history):

        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        content = str(
            message.get("content", "")
        ).strip()

        if content:
            recent_user_messages.append(content)

        # Only use the last two user messages.
        # This prevents very old mental-health discussions from affecting an unrelated new topic.
        if len(recent_user_messages) >= 2:
            break

    if not recent_user_messages:
        return False

    # -----------------------------------------
    # 5. Determine whether the recent context was mental-health related
    # -----------------------------------------

    for previous_message in recent_user_messages:

        # First use the trained classifier.
        if is_mental_health_query(previous_message):
            return True

        # Then use the conservative fallback.
        if secondary_mental_check(previous_message):
            return True

    # No evidence that this is a mental-health continuation.
    return False


# ==========================================
# AI Response Function
# ==========================================

def get_ai_response(username, user_text, history, spiritual_preference_override=None):
    reconnect_db()

    cursor.execute("""
        SELECT
            name,
            age,
            gender,
            occupation,
            goal,
            recurring_emotion,
            spiritual_preference
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
            recurring_emotion,
            spiritual_preference
            ) = profile
    else:
        (
            name,
            age,
            gender,
            occupation,
            goal,
            recurring_emotion,
            spiritual_preference
            ) = (
                "User",
                "",
                "",
                "",
                "",
                "",
                "No"
            )
    # Guest session preference overrides database/default preference
    if spiritual_preference_override is not None:
        spiritual_preference = spiritual_preference_override

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
    
    if main_confidence < 0.45:
        print(
            f"Low confidence ({main_confidence:.2f}) "
            f"→ Model predicted: {main_emotion} "
            f"→ Using fallback"
        )
        main_emotion = fallback_main_emotion(user_text)
    else:
        print(
            f"High confidence ({main_confidence:.2f}) "
            f"→ Model predicted: {main_emotion}"
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
    sub_probs = torch.softmax(sub_outputs.logits, dim=1)
    sub_pred = torch.argmax(sub_probs, dim=1).item()
    sub_confidence = sub_probs[0][sub_pred].item()

    sub_emotion = sub_encoder.inverse_transform([sub_pred])[0]

    if sub_confidence < 0.45:
        print(
            f"Low confidence ({sub_confidence:.2f}) "
            f"→ Model predicted: {sub_emotion} "
            f"→ Using sub fallback"
        )
        sub_emotion = fallback_sub_emotion(user_text, main_emotion)
    else:
        print(
            f"High confidence ({sub_confidence:.2f}) "
            f"→ Model predicted: {sub_emotion}"
        )

    # ==========================================
    # Spiritual & Mythological Retrieval
    # ==========================================

    spiritual_context = None

    if str(spiritual_preference).lower() == "yes":
        spiritual_context = retrieve_spiritual_context(
            main_emotion=main_emotion,
            sub_emotion=sub_emotion,
            user_text=user_text
        )
    del main_inputs, main_outputs
    del sub_inputs, sub_outputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # RAG Retrieval

    # Use emotion filter only if prediction is confident
    if main_confidence >= 0.45:
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
    
    distances, indices = COUNSELLING_INDEX.search(query_embedding, 5)

    best_responses = retrieve_best_counselling(search_df,indices)
    
    initial_response = generate_initial_response(
        main_emotion,
        sub_emotion,
        best_responses
        )
    rag_context = "\n\n".join(best_responses)
    print(f"Main: {main_emotion} ({main_confidence:.2f}) | Sub: {sub_emotion}")

    # Conversation Memory
    conversation = ""
    for msg in history or []:

        conversation += (f'{msg["role"]}: {msg["content"]}\n')

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

Detected Main Emotion (prediction only):
{main_emotion}

Important:
If the user's words indicate physical tiredness, sleepiness, 
overwork, lack of energy, or exhaustion, 
respond mainly as physical fatigue rather than 
sadness or depression unless the user clearly expresses emotional suffering.

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
11. Keep the response between 110 and 150 words.
12. End with one small practical action the user can take today.
13. If the query is relevant to psychological wellbeing but does not clearly express an emotion, keep:
    Main Emotion = "Not Applicable"
    Sub Emotion = "Not Applicable"
    Still provide a helpful AI response and Personalized Wellness Card.
14. Do not infer sadness, anxiety, fear, or another emotion solely from a physical symptom such as headache, fatigue, or body pain.
15. For physical wellbeing symptoms, provide general supportive guidance and recommend appropriate professional medical 
attention when the symptom is frequent, persistent, severe, or concerning.
13. Do NOT generate a Personalized Wellness Card in your reply.
14. Return only the counselling response.
The Personalized Wellness Card is generated separately by the Python function generate_wellness_card().
"""
    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
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
            main_emotion=main_emotion,
            sub_emotion=sub_emotion,
            main_confidence=main_confidence,
            sub_confidence=sub_confidence,
            user_text=user_text,
            spiritual_preference=spiritual_preference,
            spiritual_context=spiritual_context
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
    SELECT user_message, ai_reply, main_emotion, sub_emotion, created_at
    FROM (
    SELECT user_message, ai_reply, main_emotion, sub_emotion, created_at
    FROM chat_history
    WHERE username=%s
    ORDER BY id DESC
    LIMIT 20
    ) t
    ORDER BY created_at ASC
    """, (username,))

    rows = cursor.fetchall()
    messages = []

    for user_msg, ai_msg, main_emotion, sub_emotion, created_at in rows:
        messages.append({
            "role": "user",
            "content": user_msg,
            "time": str(created_at)
        })
        messages.append({
            "role": "assistant",
            "content": ai_msg,
            "main_emotion": main_emotion,
            "sub_emotion": sub_emotion,
            "time": str(created_at)
        })

    return jsonify(messages)

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    reconnect_db()

    data = request.get_json()
    username = data["username"]
    user_message = data["user_message"]

    cursor.execute("""
        DELETE FROM chat_history
        WHERE username=%s
        AND user_message=%s
        LIMIT 1
    """, (username, user_message))

    db.commit()

    return jsonify({"status": "success"})

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
            recurring_emotion,
            spiritual_preference
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
            "recurring_emotion": "",
            "spiritual_preference": "No"
        })

    return jsonify({

        "name": row[0],
        "age": row[1],
        "gender": row[2],
        "occupation": row[3],
        "goal": row[4],
        "recurring_emotion": row[5],
        "spiritual_preference": row[6]
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
            goal=%s,
            spiritual_preference=%s
        WHERE username=%s
    """,
    (
        data["name"],
        data["age"],
        data["gender"],
        data["occupation"],
        data["goal"],
        data.get("spiritual_preference", "No"),
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
    is_guest = data.get("is_guest", False)
    spiritual_preference = (
        data.get("spiritual_preference", "No")
        if is_guest
        else None
)

    if user_message == "":
        return jsonify({"reply": "Please enter a message."})

    if not is_mental_health_conversation(user_message, history):
            return jsonify({
                "main_emotion": "Not Applicable",
                "sub_emotion": "Not Applicable",
                "reply": (
                "I'm designed to help with mental health, emotions, stress, anxiety, "
                "depression, motivation, and psychological wellbeing. "
                "Please ask a question related to emotional or mental wellbeing."
                ),
                "wellness_card": ""
            })

    main_emotion, sub_emotion, reply, wellness_card = get_ai_response(
        username,
        user_message,
        history,
        spiritual_preference_override=spiritual_preference
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