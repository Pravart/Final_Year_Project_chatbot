# 🧠 AI Powered Personalized Psychological Remedies Chatbot

An AI-powered mental wellness chatbot that detects user emotions, predicts fine-grained sub-emotions, retrieves relevant counselling responses using Retrieval-Augmented Generation (RAG), and generates personalized psychological support using Large Language Models (LLMs).

The system combines Deep Learning, Natural Language Processing (NLP), FAISS similarity search, DistilBERT emotion classification, and Groq Llama 3.3 to provide personalized emotional support.

---

# 🚀 Features

## 🤖 AI Chatbot
- AI-powered psychological support
- Personalized responses using Groq Llama 3.3 (70B)
- Context-aware conversations
- Conversation history support

---

## 😊 Emotion Detection

### Main Emotion Detection
- DistilBERT Model
- 9 Main Emotions

```
Happy
Sad
Angry
Fear
Neutral
Affection
Relief
Curiosity
Embarrassment
```

### Sub Emotion Detection

- DistilBERT Model
- 28 Fine-grained Emotions

Examples:

- joy
- sadness
- anger
- disappointment
- optimism
- nervousness
- gratitude
- excitement
- pride
- remorse
- caring
- amusement
- admiration
- approval
- confusion
- realization
- surprise
- love
- etc.

---

## 🛡 Intelligent Rule-Based Fallback

When the emotion classifier confidence is low, the chatbot automatically switches to a rule-based fallback emotion detector.

Benefits:

- Better prediction for unseen sentences
- Handles mixed emotions
- Prevents incorrect high-confidence predictions
- Improves RAG retrieval quality

---

## 📚 Retrieval-Augmented Generation (RAG)

Instead of relying only on an LLM, the chatbot retrieves relevant counselling knowledge using:

- Sentence Transformers
- FAISS Vector Database
- Semantic Similarity Search

Top counselling responses are then passed to the LLM for personalized response generation.

---

## 🌱 Personalized Wellness Card

Each response also generates a personalized wellness card containing:

- 🌱 Focus
- 💨 Exercise
- 📝 Reflection
- 🎯 Tiny Goal
- 💬 Reminder

---

## 👤 User Management

- Login
- Signup
- Guest Mode
- Profile Creation
- Profile Update

---

## 📜 Chat History

- Stores every conversation
- Retrieves previous chats
- Personalized based on user history

---

## 📝 Emotion Recognition Quiz

- 10 Random Questions
- Difficulty Levels
- Rule-based Explanation System
- Score Tracking
- Accuracy Calculation
- Reset Quiz Option

---

# 🛠 Tech Stack

## Frontend

- Streamlit

## Backend

- Flask
- Flask-CORS

## Database

- MySQL

## Machine Learning

- DistilBERT
- Sentence Transformers
- Scikit-learn
- PyTorch

## Vector Database

- FAISS

## LLM

- Groq API
- Llama 3.3 70B Versatile

## Other Libraries

- Pandas
- NumPy
- Transformers
- dotenv

---

# 📂 Project Structure

```text
Final_Year_Project_chatbot/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
├── README.md
├── .env
│
├── models/
│   ├── main_emotion_model/
│   ├── sub_emotion_model/
│   ├── encoders/
│
├── datasets/
│   ├── counselling_dataset.csv
│   ├── quiz_dataset.csv
│   ├── quiz_explanations.py
│
├── screenshots/
│
└── venv/
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/Final_Year_Project_chatbot.git
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

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

## Start Flask Backend

```bash
python app.py
```

---

## Start Streamlit

```bash
streamlit run streamlit_app.py
```

---

# 📊 Model Pipeline

User Input

↓

Main Emotion Prediction (DistilBERT)

↓

Confidence Check

↓

High Confidence
→ Use Predicted Emotion

Low Confidence
→ Rule-Based Fallback

↓

Sub Emotion Prediction

↓

RAG Retrieval (FAISS)

↓

Groq Llama 3.3

↓

Personalized Response

↓

Wellness Card

---

# 🌍 Deployment

Backend

- Render

Frontend

- Streamlit Community Cloud

---

# 📷 Screenshots

- Home Page
- Login
- Signup
- Chat Interface
- Emotion Prediction
- Wellness Card
- Quiz Page
- Profile Page
- Chat History

---

# 📌 Future Scope

- Voice Interaction
- Multilingual Support
- PDF Report Generation
- Therapist Recommendation
- Mobile Application
- Fine-tuned Emotion Model
- Emotion Trend Dashboard
- Admin Analytics Panel

---

# 👨‍💻 Team Members

- Pravart Singh
- Adeeba Nizam
- Riya Pandey
- Aditya Shukla

---

# 📄 License

This project is developed as a Final Year B.Tech Project for academic purposes.