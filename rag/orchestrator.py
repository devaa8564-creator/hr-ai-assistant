"""
Step 5 - RAG Orchestrator.

Simple pipeline approach — works with ANY Ollama model (no tool-calling required):
  1. Extract employee ID from the question (if present)
  2. Fetch leave data from PostgreSQL via get_leave_info
  3. Fetch relevant policy text from ChromaDB via search_policy
  4. Send everything to Ollama in one structured prompt
  5. Return the natural language answer
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from mcp_server.server import get_leave_info, search_policy

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")


def _extract_employee_id(text: str) -> str | None:
    """Pull an employee ID like EMP001 out of free text."""
    match = re.search(r'\b(EMP\d{3,})\b', text.upper())
    return match.group(1) if match else None


def _call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Make sure it is running."
    except Exception as e:
        return f"ERROR: {e}"


def _build_prompt(question: str, leave_data: dict | None, policy_text: str) -> str:
    """Compose the full prompt with all context."""
    sections = []

    if leave_data and "error" not in leave_data:
        sections.append(
            f"EMPLOYEE LEAVE RECORD:\n"
            f"  Name: {leave_data.get('name', 'Unknown')}\n"
            f"  Employee ID: {leave_data.get('employee_id', '')}\n"
            f"  Department: {leave_data.get('department', '')}\n"
            f"  Employment type: {leave_data.get('employment_type', '')}\n"
            f"  Year: {leave_data.get('year', '')}\n"
            f"  Days taken so far: {leave_data.get('days_taken', 0)}\n"
            f"  Days pending approval: {leave_data.get('days_pending', 0)}"
        )

    if policy_text:
        sections.append(f"RELEVANT HR POLICY:\n{policy_text}")

    context = "\n\n".join(sections) if sections else "No specific employee or policy data found."

    return (
        f"You are a friendly and accurate HR assistant. "
        f"Answer the employee's question using ONLY the data provided below. "
        f"Be warm, specific, and show the calculation clearly.\n\n"
        f"{context}\n\n"
        f"EMPLOYEE QUESTION: {question}\n\n"
        f"YOUR ANSWER:"
    )


def ask(question: str, agent=None) -> str:
    """
    Main entry point. agent param is ignored (kept for UI compatibility).
    Fetches data from DB + vector store, then asks Ollama.
    """
    # 1. Try to get employee data if an ID is mentioned
    emp_id = _extract_employee_id(question)
    leave_data = None
    if emp_id:
        leave_data = get_leave_info(emp_id)

    # 2. Search policy for relevant context
    policy_text = search_policy(question)

    # 3. Build prompt and call LLM
    prompt = _build_prompt(question, leave_data, policy_text)
    return _call_ollama(prompt)


def build_agent():
    """Kept for UI compatibility — returns None (no agent object needed)."""
    # Verify Ollama is reachable
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"Warning: model '{OLLAMA_MODEL}' not found. Available: {models}")
    except Exception as e:
        print(f"Warning: Could not reach Ollama at {OLLAMA_BASE_URL}: {e}")
    return None


if __name__ == "__main__":
    print("HR AI Assistant - Pipeline Test")
    print("=" * 50)

    test_questions = [
        "Hi, I'm John Smith (EMP001). How many holiday days do I have left this year?",
        "What is the sick leave policy?",
        "How many days has Sara Jones taken? Her ID is EMP002.",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        answer = ask(q)
        print(f"A: {answer}")
        print("-" * 50)
