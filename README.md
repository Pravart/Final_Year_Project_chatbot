# 🧠 AI-Powered Personalized Psychological Remedies Chatbot

An AI-powered mental wellness chatbot that detects user emotions, predicts fine-grained sub-emotions, retrieves relevant counselling responses using Retrieval-Augmented Generation (RAG), and generates personalized psychological support using Large Language Models (LLMs).

The chatbot combines Deep Learning, Natural Language Processing (NLP), Semantic Search, FAISS, DistilBERT, ONNX optimization, Machine Learning classification, and Groq Llama 3.3 to provide intelligent and personalized emotional support.

---

# 🚀 Features

## 🤖 AI Psychological Chatbot

- AI-powered psychological counselling
- Context-aware conversations
- Personalized emotional support
- Groq Llama 3.3 (70B Versatile)
- Conversation memory

---

## 🧠 Mental Health Query Detection

Before emotion prediction, every user query is classified as:

- ✅ Mental Health Query
- ❌ Non Mental Health Query

The chatbot uses:

- Machine Learning classifier
- TF-IDF Vectorizer
- Secondary Rule-Based Verification

Non-mental queries never enter the emotion prediction pipeline and directly receive:

- Main Emotion → Not Applicable
- Sub Emotion → Not Applicable

---

# 😊 Emotion Detection

## Main Emotion Detection

DistilBERT ONNX Model

Main emotions:

- Happy
- Sad
- Angry
- Fear
- Neutral
- Affection
- Relief
- Curiosity
- Embarrassment

---

## Sub Emotion Detection

DistilBERT ONNX Model

Fine-grained emotions include:

- Joy
- Sadness
- Anger
- Nervousness
- Optimism
- Gratitude
- Pride
- Remorse
- Surprise
- Disappointment
- Caring
- Excitement
- Admiration
- Approval
- Curiosity
- Fear
- Love
- and more...

---

## 🎯 Confidence-Based Prediction

The chatbot automatically checks prediction confidence.

If confidence is high:

- Uses DistilBERT prediction

If confidence is low:

- Uses intelligent rule-based emotion fallback

This improves robustness on unseen user inputs.

---

# ⚡ ONNX Optimized Models

The DistilBERT models are converted to ONNX INT8.

Benefits:

- Faster inference
- Lower memory usage
- CPU optimized
- Reduced startup time

---

# 📚 Retrieval-Augmented Generation (RAG)

The chatbot retrieves counselling knowledge before generating responses.

Pipeline:

- Sentence Transformers
- FAISS Vector Database
- Semantic Similarity Search
- Top counselling retrieval
- Groq Llama 3.3 enhancement

This prevents hallucination and makes responses grounded in counselling knowledge.

---

# 🌱 Personalized Wellness Card

Each counselling response also generates a personalized wellness card containing:

- 🌱 Focus
- 💨 Exercise
- 📝 Reflection
- 🎯 Tiny Goal
- 💬 Reminder

---

# 👤 Personalized User Profile

The chatbot personalizes responses using:

- Name
- Age
- Gender
- Occupation
- Goal
- Previous recurring emotion

This allows emotionally personalized conversations.

---

# 📜 Chat History

The chatbot stores:

- User messages
- AI replies
- Main emotion
- Sub emotion
- Timestamp

Features:

- Refresh-safe history
- Download chat
- Delete individual chats
- Stored securely in MySQL

---

# 📝 Emotion Recognition Quiz

- 10 Random Questions
- Difficulty Levels
- Rule-Based Explanation
- Score Tracking
- Accuracy Calculation
- Reset Quiz

---

# 🛠 Tech Stack

## Frontend

- Streamlit

## Backend

- Flask

## Database

- MySQL

## Deep Learning

- DistilBERT
- ONNX Runtime
- PyTorch

## NLP

- Transformers
- Sentence Transformers

## Machine Learning

- Scikit-Learn
- TF-IDF
- Logistic Regression

## Vector Database

- FAISS

## LLM

- Groq API
- Llama 3.3 70B Versatile

## Other Libraries

- NumPy
- Pandas
- SQLAlchemy
- NLTK
- Joblib

---

# 📂 Project Structure

```text
Final_Year_Project_Chatbot/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env
│
├── models/
│   ├── distilbert_emotion_onnx/
│   ├── distilbert_sub_emotion_onnx/
│   ├── counselling_faiss.index
│   ├── counselling_embeddings.npy
│   ├── mental_classifier.pkl
│   ├── mental_vectorizer.pkl
│   ├── main_emotion_encoder.pkl
│   ├── sub_emotion_encoder.pkl
│
├── datasets/
│   ├── counselling_dataset.csv
│   ├── quiz_dataset.csv
│
├── screenshots/
│
└── venv/
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/Final_Year_Project_Chatbot.git
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY

MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=YOUR_PASSWORD
MYSQL_DATABASE=psychological_chatbot

API_URL=http://127.0.0.1:5000
```

---

# ▶ Running the Project

## Start Backend

```bash
python app.py
```

---

## Start Frontend

```bash
streamlit run streamlit_app.py
```

---

# 🧠 Model Pipeline

```text
                User Input
                     │
                     ▼
       Mental Health Classification
          │                    │
          │                    │
          ▼                    ▼
Non Mental Health        Mental Health
          │                    │
          ▼                    ▼
Return Not Applicable   Main Emotion (DistilBERT)
                               │
                               ▼
                    Confidence Verification
                               │
             ┌─────────────────┴────────────────┐
             ▼                                  ▼
      High Confidence                  Rule-Based Fallback
                               │
                               ▼
                  Sub Emotion Prediction
                               │
                               ▼
                FAISS Semantic Retrieval
                               │
                               ▼
                  Groq Llama 3.3 (70B)
                               │
                               ▼
          Personalized Psychological Reply
                               │
                               ▼
             Personalized Wellness Card
```

---

# ⚡ Performance

- DistilBERT ONNX INT8 Optimization
- Precomputed FAISS Embeddings
- CPU Optimized Inference
- Faster Semantic Search
- Reduced Startup Time
- Confidence-Based Emotion Prediction

---

# 🌍 Deployment

## Backend

- Render

## Frontend

- Streamlit Community Cloud

---

# 📷 Screenshots

- Home Page
- Login
- Signup
- Chat Interface
- Emotion Prediction
- Wellness Card
- Profile Page
- Quiz Page
- Chat History

---

# 🔮 Future Scope

- Voice-Based Counselling
- Multilingual Support
- Therapist Appointment Integration
- Emotion Trend Dashboard
- Mobile Application
- PDF Counselling Report
- Admin Analytics Dashboard

---

# 👨‍💻 Team Members

- Pravart Singh
- Adeeba Nizam
- Riya Pandey
- Aditya Shukla

---

# 📄 License

This project is developed as a Final Year B.Tech Project for academic and research purposes only.