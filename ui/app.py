"""
Step 6 - Streamlit Chat UI with live log viewer

Run:
    streamlit run ui/app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from mcp_server.server import list_employees
from rag.orchestrator import build_agent, ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR AI Assistant",
    page_icon="HR",
    layout="wide",
)

st.title("HR AI Assistant")
st.caption("Ask me anything about your leave, entitlements, or company HR policies.")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I am your HR Assistant. I can help you with:\n"
                "- Holiday / annual leave balance\n"
                "- Sick leave policy\n"
                "- Leave entitlements\n\n"
                "Just ask in plain English! (e.g. *'How many holiday days do I have left? My ID is EMP001'*)"
            ),
        }
    ]

if "all_logs" not in st.session_state:
    st.session_state.all_logs = []

if "agent_ready" not in st.session_state:
    try:
        build_agent()
        st.session_state.agent_ready = True
        st.session_state.agent_error = ""
    except Exception as e:
        st.session_state.agent_ready = False
        st.session_state.agent_error = str(e)

# ── Layout: chat on left, logs on right ───────────────────────────────────────
col_chat, col_logs = st.columns([2, 1])

# ── LEFT: Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Employee Directory")
    st.caption("Reference IDs for testing")
    try:
        employees = list_employees()
        for emp in employees:
            st.text(f"{emp['employee_id']} - {emp['name']}")
    except Exception:
        st.warning("Could not connect to database.\nRun: docker-compose up -d")

    st.divider()
    st.header("System Status")
    if st.session_state.get("agent_ready"):
        st.success("Ollama connected")
        st.info(f"Model: llama3")
    else:
        err = st.session_state.get("agent_error", "")
        st.error(f"Ollama not ready\n{err}" if err else "Ollama not reachable")

    st.divider()
    if st.button("Clear Chat + Logs"):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.all_logs = []
        st.rerun()

# ── MIDDLE: Chat ──────────────────────────────────────────────────────────────
with col_chat:
    st.subheader("Chat")

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new input
    if prompt := st.chat_input("Ask your HR question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.get("agent_ready"):
                response = "Sorry, Ollama is not available. Make sure it is running."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.spinner("Thinking..."):
                    try:
                        response, logs = ask(prompt)
                        # Prefix logs with question number
                        q_num = len([m for m in st.session_state.messages if m["role"] == "user"])
                        labelled = [f"[Q{q_num}] {entry}" for entry in logs]
                        st.session_state.all_logs.extend(labelled)
                    except Exception as e:
                        response = f"Sorry, I encountered an error: {e}"
                        st.session_state.all_logs.append(f"ERROR: {e}")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# ── RIGHT: Live Log Viewer ────────────────────────────────────────────────────
with col_logs:
    st.subheader("Pipeline Logs")
    st.caption("Live trace of each step for every question")

    if not st.session_state.all_logs:
        st.info("Logs will appear here when you ask a question.")
    else:
        # Show logs newest first, colour coded by step
        log_lines = []
        for entry in reversed(st.session_state.all_logs):
            if "Step 1" in entry:
                colour = "#4fc3f7"   # blue
            elif "Step 2" in entry:
                colour = "#81c784"   # green
            elif "Step 3" in entry:
                colour = "#ffb74d"   # orange
            elif "Step 4" in entry:
                colour = "#ce93d8"   # purple
            elif "Step 5" in entry:
                colour = "#f48fb1"   # pink
            elif "ERROR" in entry:
                colour = "#ef5350"   # red
            else:
                colour = "#b0bec5"   # grey

            log_lines.append(
                f'<div style="font-family:monospace; font-size:12px; '
                f'color:{colour}; padding:2px 0; border-bottom:1px solid #222;">'
                f'{entry}</div>'
            )

        st.markdown(
            '<div style="background:#1e1e1e; padding:10px; border-radius:8px; '
            'max-height:600px; overflow-y:auto;">'
            + "".join(log_lines)
            + "</div>",
            unsafe_allow_html=True,
        )

        # Raw log download button
        raw_logs = "\n".join(st.session_state.all_logs)
        st.download_button(
            label="Download logs as .txt",
            data=raw_logs,
            file_name="hr_assistant_logs.txt",
            mime="text/plain",
        )
