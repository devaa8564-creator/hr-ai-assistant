"""
Step 6 — Streamlit Chat UI

Run:
    streamlit run ui/app.py
"""

import os
import sys
import json
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from mcp_server.server import list_employees
from rag.orchestrator import build_agent, ask

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR AI Assistant",
    page_icon="🤝",
    layout="centered",
)

st.title("🤝 HR AI Assistant")
st.caption("Ask me anything about your leave, entitlements, or company HR policies.")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your HR Assistant. I can help you with:\n"
                "- Holiday / annual leave balance\n"
                "- Sick leave policy\n"
                "- Leave entitlements\n\n"
                "Just ask me in plain English! (e.g. *'How many holiday days do I have left? My ID is EMP001'*)"
            ),
        }
    ]

if "agent_ready" not in st.session_state:
    # Lightweight check: just verify Ollama is reachable
    try:
        build_agent()
        st.session_state.agent_ready = True
        st.session_state.agent_error = ""
    except Exception as e:
        st.session_state.agent_ready = False
        st.session_state.agent_error = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Employee Directory")
    st.caption("Reference IDs for testing")
    try:
        employees = list_employees()
        for emp in employees:
            st.text(f"{emp['employee_id']} — {emp['name']}")
    except Exception:
        st.warning("Could not connect to database.\nMake sure Docker is running:\n`docker-compose up -d`")

    st.divider()
    st.header("System Status")
    if st.session_state.get("agent_ready"):
        st.success("Ollama ready")
    else:
        err = st.session_state.get("agent_error", "")
        st.error(f"Ollama not ready:\n{err}" if err else "Ollama not reachable")
        st.info("Make sure Ollama is running and llama3 is pulled.")

    if st.button("Clear Chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your HR question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.get("agent_ready"):
            response = (
                "Sorry, the AI model is not available right now. "
                "Please make sure Ollama is running and the model is downloaded."
            )
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                try:
                    response = ask(prompt)
                except Exception as e:
                    response = f"Sorry, I encountered an error: {e}"
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
