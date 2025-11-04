import streamlit as st
from supabase import create_client, Client
import os

def get_supabase_client() -> Client:
    url = os.getenv("VITE_SUPABASE_URL")
    key = os.getenv("VITE_SUPABASE_SUPABASE_ANON_KEY")
    return create_client(url, key)

def init_session_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'session' not in st.session_state:
        st.session_state.session = None

def login_page():
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .auth-title {
        color: white;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.9);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="auth-title">Resume Analyzer</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")

            if submit:
                if email and password:
                    try:
                        supabase = get_supabase_client()
                        response = supabase.auth.sign_in_with_password({
                            "email": email,
                            "password": password
                        })

                        if response.user:
                            st.session_state.user = response.user
                            st.session_state.session = response.session

                            supabase.table('users').upsert({
                                'id': response.user.id,
                                'email': response.user.email,
                                'updated_at': 'now()'
                            }).execute()

                            st.success("Login successful!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {str(e)}")
                else:
                    st.error("Please enter both email and password")

    with tab2:
        with st.form("signup_form"):
            full_name = st.text_input("Full Name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submit = st.form_submit_button("Sign Up")

            if submit:
                if not all([full_name, email, password, confirm_password]):
                    st.error("Please fill all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    try:
                        supabase = get_supabase_client()
                        response = supabase.auth.sign_up({
                            "email": email,
                            "password": password
                        })

                        if response.user:
                            supabase.table('users').insert({
                                'id': response.user.id,
                                'email': email,
                                'full_name': full_name
                            }).execute()

                            st.success("Account created! Please login.")
                    except Exception as e:
                        st.error(f"Sign up failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

def logout():
    supabase = get_supabase_client()
    supabase.auth.sign_out()
    st.session_state.user = None
    st.session_state.session = None
    st.rerun()

def require_auth(func):
    def wrapper(*args, **kwargs):
        init_session_state()
        if st.session_state.user is None:
            login_page()
            return None
        return func(*args, **kwargs)
    return wrapper
