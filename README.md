# 🧠 Psychological Remedies Chatbot

An AI-powered Psychological Remedies Chatbot that detects emotions, predicts sub-emotions, retrieves relevant counseling knowledge using RAG, and provides personalized responses.

---

## Features

- Emotion Detection (9 Main Emotions)
- Sub Emotion Prediction (28 Sub Emotions)
- Retrieval-Augmented Generation (RAG)
- Personalized AI Responses
- User Login & Signup
- Guest Mode
- Profile Management
- Chat History Storage
- MySQL Database Integration
- Groq LLM Integration
- Streamlit Frontend
- Flask Backend

---

## Tech Stack

Frontend
- Streamlit

Backend
- Flask
- Flask-CORS

Database
- MySQL

Machine Learning
- Scikit-learn
- Sentence Transformers
- FAISS

LLM
- Groq API (Llama 3.3 70B)

---

## Project Structure

```
Final_Year_Project/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .env
├── models/
├── dataset/
├── screenshots/
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone <your-github-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GROQ_API_KEY=YOUR_API_KEY

MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=YOUR_PASSWORD
MYSQL_DATABASE=psychological_chatbot

API_URL=http://127.0.0.1:5000
```

Run Flask

```bash
python app.py
```

Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## Screenshots

- Home Page
- Login
- Signup
- Guest Mode
- Chat Interface
- Emotion Prediction
- Profile Page
- Chat History

---

## Future Scope

- Voice Interaction
- Multilingual Support
- Therapist Recommendation
- PDF Report Generation
- Mobile Application
- Cloud Deployment

---

## Author

Pravart Singh
Adeeba Nizam
Riya Pandey
Aditya Shukla
