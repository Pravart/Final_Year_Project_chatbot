import os
import hashlib
import uuid
from typing import Any

import mysql.connector
import requests
import streamlit as st
from dotenv import load_dotenv
from mysql.connector import Error as MySQLError
from requests import RequestException

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000").rstrip("/")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "psychological_chatbot")

GENDER_OPTIONS = ["Male", "Female", "Other"]

st.set_page_config(
    page_title="Psychological Remedies Chatbot",
    page_icon="🧠",
    layout="wide",
)


def hash_password(password: str) -> str:
    """Return a SHA-256 hash so raw passwords are not stored in MySQL."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_db_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=False,
    )


def init_state() -> None:
    defaults: dict[str, Any] = {
        "screen": "welcome",
        "authenticated": False,
        "is_guest": False,
        "username": None,
        "messages": [],
        "navigation": "💬 Chat",
        "guest_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def go_to(screen: str) -> None:
    st.session_state.screen = screen
    st.rerun()


def load_history(username: str) -> list[dict[str, str]]:
    try:
        response = requests.get(
            f"{API_URL}/history",
            params={"username": username},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except (RequestException, ValueError):
        st.warning("Previous chat history could not be loaded.")
        return []


def fetch_profile(username: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{API_URL}/profile",
            params={"username": username},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "name": data.get("name") or "",
            "age": int(data.get("age") or 18),
            "gender": data.get("gender") or "Other",
            "occupation": data.get("occupation") or "",
            "goal": data.get("goal") or "",
        }
    except (RequestException, ValueError, TypeError):
        st.error("Profile could not be loaded.")
        return {
            "name": "",
            "age": 18,
            "gender": "Other",
            "occupation": "",
            "goal": "",
        }


def update_profile(username: str, profile: dict[str, Any]) -> bool:
    try:
        response = requests.post(
            f"{API_URL}/update_profile",
            json={"username": username, **profile},
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("status") == "success"
    except (RequestException, ValueError):
        return False


def register_user(
    username: str,
    password: str,
    name: str,
    age: int,
    gender: str,
    occupation: str,
) -> tuple[bool, str]:
    if not username.strip() or not password:
        return False, "Username and password are required."

    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO users (username, password)
            VALUES (%s, %s)
            """,
            (username.strip(), hash_password(password)),
        )

        cursor.execute(
            """
            INSERT INTO user_profile
            (username, name, age, gender, occupation)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                username.strip(),
                name.strip(),
                int(age),
                gender,
                occupation.strip(),
            ),
        )

        connection.commit()
        return True, "Account created successfully. Please log in."

    except MySQLError as exc:
        if connection:
            connection.rollback()

        if getattr(exc, "errno", None) == 1062:
            return False, "That username already exists."

        return False, f"Account creation failed: {exc}"

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def authenticate_user(username: str, password: str) -> bool:
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT password
            FROM users
            WHERE username = %s
            """,
            (username.strip(),),
        )
        row = cursor.fetchone()

        if not row:
            return False

        stored_password = row[0]
        entered_hash = hash_password(password)

        # Supports new hashed accounts and older plain-text test accounts.
        return stored_password == entered_hash or stored_password == password

    except MySQLError as exc:
        st.exception(exc)
        st.error("Database login error.")
        return False

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def render_header() -> None:
    st.title("🧠 Psychological Remedies Chatbot")
    st.caption(
        "Emotion-aware, RAG-supported guidance with optional personalized memory."
    )


def render_welcome() -> None:
    render_header()
    st.write("Talk freely. Your privacy choices come first.")

    guest_col, login_col, signup_col = st.columns(3)

    with guest_col:
        if st.button(
            "👤 Continue as Guest",
            use_container_width=True,
            type="primary",
        ):
            guest_id = f"guest_{uuid.uuid4().hex[:12]}"
            st.session_state.guest_id = guest_id
            st.session_state.username = guest_id
            st.session_state.is_guest = True
            st.session_state.authenticated = True
            st.session_state.messages = []
            st.session_state.screen = "app"
            st.rerun()

    with login_col:
        if st.button("🔑 Login", use_container_width=True):
            go_to("login")

    with signup_col:
        if st.button("📝 Sign Up", use_container_width=True):
            go_to("signup")

    st.info(
        "Guest mode uses only temporary on-screen memory. "
        "A registered account can load previous chats and profile details."
    )


def render_login() -> None:
    render_header()
    st.subheader("🔑 Login")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Login",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if authenticate_user(username, password):
            st.session_state.username = username.strip()
            st.session_state.authenticated = True
            st.session_state.is_guest = False
            st.session_state.messages = load_history(username.strip())
            st.session_state.screen = "app"
            st.rerun()
        else:
            st.error("Invalid username or password.")

    if st.button("← Back"):
        go_to("welcome")


def render_signup() -> None:
    render_header()
    st.subheader("📝 Create Account")

    with st.form("signup_form", clear_on_submit=False):
        username = st.text_input("Choose Username")
        password = st.text_input("Choose Password", type="password")
        name = st.text_input("Full Name")
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=18,
            step=1,
        )
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        occupation = st.text_input("Occupation")
        submitted = st.form_submit_button(
            "Create Account",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        success, message = register_user(
            username=username,
            password=password,
            name=name,
            age=int(age),
            gender=gender,
            occupation=occupation,
        )
        if success:
            st.success(message)
            st.session_state.screen = "login"
            st.rerun()
        else:
            st.error(message)

    if st.button("← Back"):
        go_to("welcome")


def render_sidebar() -> None:
    st.sidebar.title("🧠 Psychological Remedies")

    if st.session_state.is_guest:
        st.sidebar.info("Guest session")
        st.session_state.navigation = "💬 Chat"
    else:
        st.sidebar.success(f"Signed in as {st.session_state.username}")

        st.session_state.navigation = st.sidebar.radio(
            "Navigation",
            ["💬 Chat", "👤 Profile"],
            key="navigation_radio",
        )
    st.sidebar.markdown("### 🕒 Recent Chats")

    recent_users = [
        msg["content"][:35] + ("..." if len(msg["content"]) > 35 else "")
        for msg in st.session_state.messages
        if msg["role"] == "user"
    ][-5:]

    if recent_users:
        for chat in recent_users:
            st.sidebar.write("• " + chat)
    else:
        st.sidebar.write("No chat history yet.")

    if st.sidebar.button("🧹 Clear visible chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        reset_session()

    st.sidebar.caption(
        "This chatbot provides supportive information and is not a substitute "
        "for professional diagnosis, therapy, or emergency services."
    )


def render_profile_page() -> None:
    username = st.session_state.username
    profile = fetch_profile(username)

    st.header("👤 My Profile")

    gender = profile["gender"]
    if gender not in GENDER_OPTIONS:
        gender = "Other"

    with st.form("profile_form"):
        name = st.text_input("Name", value=profile["name"])
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=int(profile["age"]),
            step=1,
        )
        gender = st.selectbox(
            "Gender",
            GENDER_OPTIONS,
            index=GENDER_OPTIONS.index(gender),
        )
        occupation = st.text_input(
            "Occupation",
            value=profile["occupation"],
        )
        goal = st.text_area(
            "Goal",
            value=profile["goal"],
            height=120,
        )
        submitted = st.form_submit_button(
            "Update Profile",
            type="primary",
        )

    if submitted:
        updated = update_profile(
            username,
            {
                "name": name.strip(),
                "age": int(age),
                "gender": gender,
                "occupation": occupation.strip(),
                "goal": goal.strip(),
            },
        )
        if updated:
            st.success("Profile updated successfully.")
        else:
            st.error("Profile update failed. Check that Flask is running.")


def send_chat_message(prompt: str) -> dict[str, str] | None:
    username = st.session_state.username
    history = st.session_state.messages[-10:]

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "username": username,
                "message": prompt,
                "history": history,
                "is_guest": st.session_state.is_guest,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        required = {"main_emotion", "sub_emotion", "reply"}
        if not required.issubset(result):
            raise ValueError("Incomplete backend response.")

        return result

    except RequestException as exc:
        st.error(f"Could not reach the Flask backend: {exc}")
    except ValueError:
        st.error("The backend returned an invalid response.")

    return None


def render_chat_page() -> None:
    st.header("💬 Supportive Chat")
    st.write("Talk freely. I'm here to listen and offer practical support.")

    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(
                "Hello. Share what has been on your mind, and I will respond "
                "with supportive, personalized guidance."
            )

    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        avatar = "👤" if role == "user" else "🧠"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.get("content", ""))

    prompt = st.chat_input("Type your message...")

    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        result = send_chat_message(prompt)

    if result is None:
        return

    main_emotion = result["main_emotion"]
    sub_emotion = result["sub_emotion"]
    reply = result["reply"]

    assistant_content = (
        f"**😊 Main emotion:** {main_emotion}\n\n"
        f"**🎯 Sub-emotion:** {sub_emotion}\n\n"
        f"{reply}"
    )
    # Refresh chat history from database
    st.session_state.messages = load_history(st.session_state.username)
    # # If latest assistant message is not yet in history, append it temporarily
    if (
        not st.session_state.messages or
        st.session_state.messages[-1]["content"] != assistant_content
    ):
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )
    st.rerun()


def render_app() -> None:
    render_sidebar()
    render_header()

    if (
        not st.session_state.is_guest
        and st.session_state.navigation == "👤 Profile"
    ):
        render_profile_page()
    else:
        render_chat_page()

init_state()

if not st.session_state.authenticated:
    if st.session_state.screen == "login":
        render_login()
    elif st.session_state.screen == "signup":
        render_signup()
    else:
        render_welcome()
else:
    render_app()