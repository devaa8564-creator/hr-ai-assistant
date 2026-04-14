"""
Step 5 - RAG Orchestrator with deep debug logging.
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
from mcp_server.client import call_tool, list_tools, ping

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
MCP_SERVER_URL  = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hr_assistant")


# ── Step 1: Extract employee ID ───────────────────────────────────────────────

def _extract_employee_id(text: str, logs: list) -> str | None:
    pattern = r'\b(EMP\d{3,})\b'
    logs.append(("INFO",  "STEP 1 — Extract Employee ID"))
    logs.append(("DEBUG", f"  Input text   : {text}"))
    logs.append(("DEBUG", f"  Regex pattern: {pattern}"))

    match = re.search(pattern, text.upper())
    if match:
        logs.append(("OK",    f"  Match found  : {match.group(1)}  (position {match.start()}-{match.end()})"))
        return match.group(1)
    else:
        logs.append(("WARN",  "  No match     : No employee ID found in question"))
        return None


# ── Step 2: Call MCP tool -> get_leave_info ───────────────────────────────────

def _get_leave_info(employee_id: str, logs: list) -> dict:
    logs.append(("INFO",  "STEP 2 — MCP Tool Call: get_leave_info"))
    logs.append(("DEBUG", f"  MCP Server   : {MCP_SERVER_URL}"))
    logs.append(("DEBUG",  "  Protocol     : SSE (Server-Sent Events)"))
    logs.append(("DEBUG",  "  MCP method   : tools/call"))
    logs.append(("DEBUG", f"  Tool name    : get_leave_info"))
    logs.append(("DEBUG", f"  Arguments    : {{ \"employee_id\": \"{employee_id}\" }}"))

    try:
        t0 = time.time()
        result = call_tool("get_leave_info", {"employee_id": employee_id})
        elapsed = time.time() - t0

        logs.append(("DEBUG", f"  Round-trip   : {elapsed:.3f}s"))
        logs.append(("DEBUG", f"  MCP response : {result}"))

        if not result or "error" in result:
            logs.append(("WARN",  f"  Tool error   : {result.get('error') if result else 'No response'}"))
            return result or {"error": "No response from MCP server"}

        logs.append(("OK",   f"  name         : {result.get('name')}"))
        logs.append(("OK",   f"  department   : {result.get('department')}"))
        logs.append(("OK",   f"  employment   : {result.get('employment_type')}"))
        logs.append(("OK",   f"  year         : {result.get('year')}"))
        logs.append(("OK",   f"  days_taken   : {result.get('days_taken')}"))
        logs.append(("OK",   f"  days_pending : {result.get('days_pending')}"))
        return result

    except Exception as e:
        logs.append(("ERROR", f"  MCP call failed: {e}"))
        return {"error": str(e)}


# ── Step 3: Call MCP tool -> search_policy ────────────────────────────────────

def _search_policy(query: str, logs: list) -> str:
    logs.append(("INFO",  "STEP 3 — MCP Tool Call: search_policy"))
    logs.append(("DEBUG", f"  MCP Server   : {MCP_SERVER_URL}"))
    logs.append(("DEBUG",  "  Protocol     : SSE (Server-Sent Events)"))
    logs.append(("DEBUG",  "  MCP method   : tools/call"))
    logs.append(("DEBUG",  "  Tool name    : search_policy"))
    logs.append(("DEBUG", f"  Arguments    : {{ \"query\": \"{query[:60]}...\" }}"))
    logs.append(("DEBUG",  "  Server executes: embed query -> ChromaDB cosine search -> top 3 chunks"))

    try:
        t0 = time.time()
        result = call_tool("search_policy", {"query": query})
        elapsed = time.time() - t0

        logs.append(("DEBUG", f"  Round-trip   : {elapsed:.3f}s"))

        if not result:
            logs.append(("WARN",  "  No policy text returned"))
            return "No relevant policy found."

        preview = str(result)[:120].replace("\n", " ")
        logs.append(("OK",   f"  Policy text  : {len(str(result))} characters"))
        logs.append(("OK",   f"  Preview      : \"{preview}...\""))
        return result

    except Exception as e:
        logs.append(("ERROR", f"  MCP call failed: {e}"))
        return f"Policy search error: {e}"


# ── Step 4: Build prompt ──────────────────────────────────────────────────────

def _build_prompt(question: str, leave_data: dict | None, policy_text: str, logs: list) -> str:
    logs.append(("INFO",  "STEP 4 — Build Prompt"))

    sections = []
    if leave_data and "error" not in leave_data:
        employee_block = (
            f"EMPLOYEE LEAVE RECORD:\n"
            f"  Name: {leave_data.get('name')}\n"
            f"  Employee ID: {leave_data.get('employee_id')}\n"
            f"  Department: {leave_data.get('department')}\n"
            f"  Employment type: {leave_data.get('employment_type')}\n"
            f"  Year: {leave_data.get('year')}\n"
            f"  Days taken so far: {leave_data.get('days_taken', 0)}\n"
            f"  Days pending approval: {leave_data.get('days_pending', 0)}"
        )
        sections.append(employee_block)
        logs.append(("DEBUG", "  Block added  : EMPLOYEE LEAVE RECORD"))
    else:
        logs.append(("WARN",  "  No employee block (no ID or DB error)"))

    if policy_text:
        sections.append(f"RELEVANT HR POLICY:\n{policy_text}")
        logs.append(("DEBUG", "  Block added  : RELEVANT HR POLICY"))

    context = "\n\n".join(sections) if sections else "No specific data found."

    system_instruction = (
        "You are a friendly and accurate HR assistant. "
        "Answer the employee's question using ONLY the data provided below. "
        "Be warm, specific, and show the calculation clearly."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"{context}\n\n"
        f"EMPLOYEE QUESTION: {question}\n\n"
        f"YOUR ANSWER:"
    )

    logs.append(("DEBUG", f"  System instr : {len(system_instruction)} characters"))
    logs.append(("DEBUG", f"  Context block: {len(context)} characters"))
    logs.append(("DEBUG", f"  Full prompt  : {len(prompt)} characters total"))
    logs.append(("OK",    f"  Prompt preview (first 200 chars):"))
    logs.append(("CODE",   prompt[:200].replace("\n", " | ")))

    return prompt


# ── Step 5: Call Ollama ───────────────────────────────────────────────────────

def _call_ollama(prompt: str, logs: list) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    logs.append(("INFO",  "STEP 5 — Call Ollama LLM"))
    logs.append(("DEBUG", f"  Endpoint     : POST {url}"))
    logs.append(("DEBUG", f"  Model        : {payload['model']}"))
    logs.append(("DEBUG", f"  Stream       : {payload['stream']}"))
    logs.append(("DEBUG", f"  Prompt length: {len(prompt)} characters"))
    logs.append(("DEBUG",  "  Waiting for response ..."))

    try:
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - t0

        logs.append(("DEBUG", f"  HTTP status  : {resp.status_code}"))
        logs.append(("DEBUG", f"  Response time: {elapsed:.2f}s"))

        resp.raise_for_status()
        data = resp.json()

        answer = data.get("response", "").strip()
        total_dur = data.get("total_duration", 0) / 1e9
        eval_count = data.get("eval_count", "?")
        prompt_eval = data.get("prompt_eval_count", "?")

        logs.append(("OK",   f"  Model done   : true"))
        logs.append(("OK",   f"  Total duration : {total_dur:.2f}s"))
        logs.append(("OK",   f"  Prompt tokens  : {prompt_eval}"))
        logs.append(("OK",   f"  Answer tokens  : {eval_count}"))
        logs.append(("OK",   f"  Answer length  : {len(answer)} characters"))
        logs.append(("CODE",  f"Answer preview: {answer[:200]}"))

        return answer

    except requests.exceptions.ConnectionError:
        logs.append(("ERROR", f"  Connection refused at {url}"))
        return "ERROR: Cannot connect to Ollama. Make sure it is running."
    except Exception as e:
        logs.append(("ERROR", f"  Ollama error : {e}"))
        return f"ERROR: {e}"


# ── Main entry point ──────────────────────────────────────────────────────────

def ask(question: str, agent=None) -> tuple[str, list]:
    """
    Main entry point. Returns (answer, logs) tuple.
    logs is a list of (level, message) tuples.
    """
    logs = []

    t_total = time.time()
    logs.append(("INFO",  "=" * 55))
    logs.append(("INFO",  "HR ASSISTANT — DEBUG TRACE"))
    logs.append(("INFO",  "=" * 55))
    logs.append(("DEBUG", f"  Question     : {question}"))
    logs.append(("DEBUG", f"  Timestamp    : {time.strftime('%Y-%m-%d %H:%M:%S')}"))
    logs.append(("INFO",  "-" * 55))

    emp_id     = _extract_employee_id(question, logs)
    logs.append(("INFO",  "-" * 55))

    leave_data = _get_leave_info(emp_id, logs) if emp_id else None
    if not emp_id:
        logs.append(("INFO",  "STEP 2 — Skipped (no employee ID in question)"))
    logs.append(("INFO",  "-" * 55))

    policy_text = _search_policy(question, logs)
    logs.append(("INFO",  "-" * 55))

    prompt = _build_prompt(question, leave_data, policy_text, logs)
    logs.append(("INFO",  "-" * 55))

    answer = _call_ollama(prompt, logs)
    logs.append(("INFO",  "-" * 55))

    total = time.time() - t_total
    logs.append(("OK",   f"TOTAL TIME     : {total:.2f}s"))
    logs.append(("INFO",  "=" * 55))

    return answer, logs


def build_agent():
    """
    Checks both MCP server and Ollama are reachable.
    Raises RuntimeError if either is down.
    """
    # Check MCP server
    if not ping():
        raise RuntimeError(
            f"MCP server not reachable at {MCP_SERVER_URL}\n"
            "Start it with:  python mcp_server/server.py"
        )
    tools = list_tools()
    logger.info("MCP server ready — tools: %s", [t['name'] for t in tools])

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            raise RuntimeError(f"Model '{OLLAMA_MODEL}' not found in Ollama. Run: ollama pull {OLLAMA_MODEL}")
        logger.info("Ollama ready — model: %s", OLLAMA_MODEL)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_BASE_URL}")

    return None


if __name__ == "__main__":
    print("HR AI Assistant - Debug Pipeline Test")
    print("=" * 55)
    q = "How many holiday days do I have left? My ID is EMP001"
    answer, logs = ask(q)
    for level, msg in logs:
        print(f"[{level:<5}] {msg}")
    print(f"\nANSWER:\n{answer}")
