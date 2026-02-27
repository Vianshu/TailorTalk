"""
Streamlit Frontend for TailorTalk — Titanic Chat Agent.

A beautiful, interactive chat interface for exploring the Titanic dataset
using natural language queries.
"""

import streamlit as st
import base64
import time
import pandas as pd
import re
import sys
import os

# Add parent directory to path to access backend module directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.agent import run_agent_query

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TailorTalk — Titanic Dataset Chat Agent",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; background-color: #ffffff; color: #111111; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* Center the main container aggressively */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
        max-width: 700px;
    }

    /* Minimalist Header */
    .hero-title {
        font-size: 1.4rem;
        font-weight: 500;
        color: #111111;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #888888;
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Chat Messages */
    .user-message {
        background: #f4f4f4;
        color: #111111;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .bot-message {
        background: transparent;
        color: #111111;
        padding: 0.8rem 0;
        margin: 1rem 0;
        max-width: 100%;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Visualization container */
    .viz-container { 
        border-radius: 8px; 
        padding: 0; 
        margin: 1rem 0;
        overflow: hidden;
        border: 1px solid #eeeeee; 
    }

    /* Input Box */
    .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
    }
    .stTextInput>div>div>input:focus { border-color: #bbbbbb !important; box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important; }
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

def send_message(message: str) -> dict:
    """Process a message directly through the LangChain agent without relying on FastAPI."""
    try:
        return run_agent_query(message)
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "visualization": None,
        }




def format_message(text: str) -> str:
    """Format basic markdown (bold, italic, newlines) into HTML since it's injected inside a div."""
    if not text:
        return ""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = text.replace('\n', '<br>')
    return text





st.markdown("""
<div class="hero-title">TailorTalk</div>
<div class="hero-subtitle">Titanic Dataset Agent</div>
""", unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        pass

    for msg in st.session_state.messages:
        content_html = format_message(msg["content"])
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{content_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{content_html}</div>', unsafe_allow_html=True)
            if msg.get("visualization"):
                img_bytes = base64.b64decode(msg["visualization"])
                st.markdown('<div class="viz-container">', unsafe_allow_html=True)
                st.image(img_bytes, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


if "pending_question" in st.session_state:
    pending = st.session_state.pending_question
    del st.session_state.pending_question

    # Add user message
    st.session_state.messages.append({"role": "user", "content": pending})

    # Get response
    with st.status("Processing...", expanded=True) as status:
        st.write("Connecting to agent...")
        response = send_message(pending)
        status.update(label="Complete", state="complete", expanded=False)

    # Add bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "visualization": response.get("visualization"),
    })
    st.rerun()

# Chat input
user_input = st.chat_input("Ask a question about the dataset...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from backend
    with st.status("Processing...", expanded=True) as status:
        st.write("Connecting to agent...")
        response = send_message(user_input)
        status.update(label="Complete", state="complete", expanded=False)

    # Add bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "visualization": response.get("visualization"),
    })
    st.rerun()
