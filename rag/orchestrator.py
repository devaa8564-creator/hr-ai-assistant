"""
Step 5 - RAG Orchestrator.

Simple pipeline approach - works with ANY Ollama model (no tool-calling required):
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
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from mcp_server.server import get_leave_info, search_policy

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")

# ── Logger setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hr_assistant")


def _extract_employee_id(text: str) -> str | None:
    """Pull an employee ID like EMP001 out of free text."""
    match = re.search(r'\b(EMP\d{3,})\b', text.upper())
    return match.group(1) if match else None


def _call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        logger.info("Sending prompt to Ollama (model: %s) ...", OLLAMA_MODEL)
        t0 = time.time()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        elapsed = time.time() - t0
        data = resp.json()
        logger.info("Ollama responded in %.1fs", elapsed)
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama at %s", OLLAMA_BASE_URL)
        return "ERROR: Cannot connect to Ollama. Make sure it is running."
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
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


def ask(question: str, agent=None) -> tuple[str, list[str]]:
    """
    Main entry point. Returns (answer, logs) tuple.
    agent param is ignored (kept for UI compatibility).
    """
    logs = []

    def log(msg: str):
        logger.info(msg)
        logs.append(msg)

    log(f"Question received: {question}")

    # Step 1 - Extract employee ID
    emp_id = _extract_employee_id(question)
    if emp_id:
        log(f"Step 1: Employee ID detected -> {emp_id}")
    else:
        log("Step 1: No employee ID found in question")

    # Step 2 - Fetch leave data
    leave_data = None
    if emp_id:
        log(f"Step 2: Querying PostgreSQL for {emp_id} ...")
        t0 = time.time()
        leave_data = get_leave_info(emp_id)
        elapsed = time.time() - t0
        if "error" in leave_data:
            log(f"Step 2: DB error -> {leave_data['error']}")
        else:
            log(f"Step 2: DB result ({elapsed:.2f}s) -> {leave_data['name']}, "
                f"days_taken={leave_data['days_taken']}, days_pending={leave_data['days_pending']}")
    else:
        log("Step 2: Skipped DB query (no employee ID)")

    # Step 3 - Search policy
    log("Step 3: Searching ChromaDB for relevant policy ...")
    t0 = time.time()
    policy_text = search_policy(question)
    elapsed = time.time() - t0
    preview = policy_text[:120].replace("\n", " ") + "..." if len(policy_text) > 120 else policy_text
    log(f"Step 3: Policy found ({elapsed:.2f}s) -> \"{preview}\"")

    # Step 4 - Build prompt
    prompt = _build_prompt(question, leave_data, policy_text)
    log(f"Step 4: Prompt built ({len(prompt)} characters)")

    # Step 5 - Call Ollama
    log(f"Step 5: Calling Ollama ({OLLAMA_MODEL}) ...")
    t0 = time.time()
    answer = _call_ollama(prompt)
    elapsed = time.time() - t0
    log(f"Step 5: Ollama responded in {elapsed:.1f}s ({len(answer)} characters)")

    return answer, logs


def build_agent():
    """Kept for UI compatibility - returns None (no agent object needed)."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            logger.warning("Model '%s' not found. Available: %s", OLLAMA_MODEL, models)
    except Exception as e:
        logger.warning("Could not reach Ollama at %s: %s", OLLAMA_BASE_URL, e)
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
        answer, logs = ask(q)
        print("\nLogs:")
        for entry in logs:
            print(f"  {entry}")
        print(f"\nA: {answer}")
        print("-" * 50)
