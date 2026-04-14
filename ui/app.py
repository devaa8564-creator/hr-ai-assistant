"""
Step 6 - Streamlit Chat UI with deep debug log viewer
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from mcp_server.client import call_tool, ping
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
                "Just ask in plain English!\n"
                "e.g. *'How many holiday days do I have left? My ID is EMP001'*"
            ),
        }
    ]

if "all_logs" not in st.session_state:
    st.session_state.all_logs = []   # list of (q_num, level, message)

if "agent_ready" not in st.session_state:
    try:
        build_agent()
        st.session_state.agent_ready = True
        st.session_state.agent_error = ""
    except Exception as e:
        st.session_state.agent_ready = False
        st.session_state.agent_error = str(e)

# ── Colour map for log levels ─────────────────────────────────────────────────
LEVEL_STYLE = {
    "INFO":  {"color": "#90caf9", "prefix": "INFO "},   # light blue
    "DEBUG": {"color": "#b0bec5", "prefix": "DBG  "},   # grey
    "OK":    {"color": "#a5d6a7", "prefix": "OK   "},   # green
    "WARN":  {"color": "#ffcc80", "prefix": "WARN "},   # amber
    "ERROR": {"color": "#ef9a9a", "prefix": "ERR  "},   # red
    "CODE":  {"color": "#ce93d8", "prefix": "     "},   # purple (code/preview)
}

def render_logs(logs):
    """Render (q_num, level, message) tuples as a styled HTML debug console."""
    lines = []
    for q_num, level, msg in logs:
        style = LEVEL_STYLE.get(level, LEVEL_STYLE["DEBUG"])
        colour  = style["color"]
        prefix  = style["prefix"]
        q_label = f"<span style='color:#546e7a;'>[Q{q_num}]</span> " if q_num else ""
        escaped = msg.replace("<", "&lt;").replace(">", "&gt;")
        lines.append(
            f'<div style="font-family:Consolas,monospace; font-size:11.5px; '
            f'color:{colour}; padding:1px 4px; white-space:pre-wrap; word-break:break-all;">'
            f'{q_label}'
            f'<span style="color:#546e7a;">{prefix}</span>'
            f'{escaped}'
            f'</div>'
        )
    return (
        '<div style="background:#0d1117; padding:10px 6px; border-radius:8px; '
        'max-height:650px; overflow-y:auto; border:1px solid #30363d;">'
        + "".join(lines)
        + "</div>"
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Employee Directory")
    try:
        employees = call_tool("list_employees", {})
        for emp in (employees or []):
            st.text(f"{emp['employee_id']} - {emp['name']}")
    except Exception:
        st.warning("MCP server not reachable.\nRun: python mcp_server/server.py")

    st.divider()
    st.header("System Status")
    mcp_ok = ping()
    if mcp_ok:
        st.success("MCP Server connected")
        st.caption("http://localhost:8001/sse")
    else:
        st.error("MCP Server offline")
        st.caption("Run: python mcp_server/server.py")

    if st.session_state.get("agent_ready"):
        st.success("Ollama connected")
        st.caption("Model: llama3")
    else:
        st.error("Ollama not reachable")

    st.divider()
    if st.button("Clear Chat + Logs"):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.all_logs = []
        st.rerun()

# ── Main layout: chat left, logs right ───────────────────────────────────────
col_chat, col_logs = st.columns([2, 1])

# ── Chat column ───────────────────────────────────────────────────────────────
with col_chat:
    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                        response, raw_logs = ask(prompt)
                        q_num = len([m for m in st.session_state.messages
                                     if m["role"] == "user"])
                        for level, msg_text in raw_logs:
                            st.session_state.all_logs.append((q_num, level, msg_text))
                    except Exception as e:
                        response = f"Sorry, I encountered an error: {e}"
                        q_num = len([m for m in st.session_state.messages
                                     if m["role"] == "user"])
                        st.session_state.all_logs.append((q_num, "ERROR", str(e)))

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# ── Debug log column ──────────────────────────────────────────────────────────
with col_logs:
    st.subheader("Debug Trace")

    # Legend
    legend_html = " &nbsp; ".join(
        f'<span style="color:{v["color"]}; font-family:monospace; font-size:11px;">'
        f'{v["prefix"].strip() or "CODE"}</span>'
        for v in LEVEL_STYLE.values()
    )
    st.markdown(f"<small>{legend_html}</small>", unsafe_allow_html=True)

    if not st.session_state.all_logs:
        st.info("Ask a question to see the full debug trace here.")
    else:
        # Filter controls
        show_debug = st.checkbox("Show DEBUG lines", value=True)
        show_code  = st.checkbox("Show CODE previews", value=True)

        filtered = [
            (q, lvl, msg) for q, lvl, msg in reversed(st.session_state.all_logs)
            if not (lvl == "DEBUG" and not show_debug)
            if not (lvl == "CODE"  and not show_code)
        ]

        st.markdown(render_logs(filtered), unsafe_allow_html=True)

        # Download
        raw_text = "\n".join(
            f"[Q{q}] [{lvl}] {msg}"
            for q, lvl, msg in st.session_state.all_logs
        )
        st.download_button(
            label="Download full debug log",
            data=raw_text,
            file_name="hr_debug_log.txt",
            mime="text/plain",
        )
