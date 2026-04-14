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
import psycopg2
import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://hruser:hrpassword@localhost:5432/hrdb")
CHROMA_HOST     = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hr_policy")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hr_assistant")

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


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


# ── Step 2: Query PostgreSQL ──────────────────────────────────────────────────

def _get_leave_info(employee_id: str, logs: list) -> dict:
    sql = """
    SELECT e.name, e.department, e.employment_type,
           l.year, l.days_taken, l.days_pending
    FROM employees e
    LEFT JOIN employee_leave l
        ON e.employee_id = l.employee_id
        AND l.year = EXTRACT(YEAR FROM CURRENT_DATE)
    WHERE e.employee_id = %s
    """
    params = (employee_id.upper(),)

    logs.append(("INFO",  "STEP 2 — Query PostgreSQL"))
    logs.append(("DEBUG", f"  Host         : {DATABASE_URL.split('@')[-1]}"))
    logs.append(("DEBUG", f"  SQL          : SELECT name, department, employment_type,"))
    logs.append(("DEBUG",  "                         year, days_taken, days_pending"))
    logs.append(("DEBUG",  "                 FROM employees e"))
    logs.append(("DEBUG",  "                 LEFT JOIN employee_leave l ON e.employee_id = l.employee_id"))
    logs.append(("DEBUG",  "                 AND l.year = EXTRACT(YEAR FROM CURRENT_DATE)"))
    logs.append(("DEBUG",  "                 WHERE e.employee_id = %s"))
    logs.append(("DEBUG", f"  Params       : {params}"))

    try:
        t0 = time.time()
        conn = psycopg2.connect(DATABASE_URL)
        logs.append(("OK",   f"  DB connected  in {time.time()-t0:.3f}s"))

        cur = conn.cursor()
        t1 = time.time()
        cur.execute(sql, params)
        row = cur.fetchone()
        elapsed = time.time() - t1
        conn.close()

        logs.append(("DEBUG", f"  Query executed in {elapsed:.3f}s"))
        logs.append(("DEBUG", f"  Raw row      : {row}"))

        if not row:
            logs.append(("WARN",  f"  Result       : No employee found for ID={employee_id}"))
            return {"error": f"No employee found with ID {employee_id}"}

        name, dept, emp_type, year, days_taken, days_pending = row
        result = {
            "employee_id":     employee_id.upper(),
            "name":            name,
            "department":      dept,
            "employment_type": emp_type,
            "year":            year,
            "days_taken":      days_taken or 0,
            "days_pending":    days_pending or 0,
        }
        logs.append(("OK",   f"  name         : {result['name']}"))
        logs.append(("OK",   f"  department   : {result['department']}"))
        logs.append(("OK",   f"  emp_type     : {result['employment_type']}"))
        logs.append(("OK",   f"  year         : {result['year']}"))
        logs.append(("OK",   f"  days_taken   : {result['days_taken']}"))
        logs.append(("OK",   f"  days_pending : {result['days_pending']}"))
        return result

    except Exception as e:
        logs.append(("ERROR", f"  DB error     : {e}"))
        return {"error": str(e)}


# ── Step 3: Query ChromaDB ────────────────────────────────────────────────────

def _search_policy(query: str, logs: list) -> str:
    logs.append(("INFO",  "STEP 3 — Query ChromaDB (Vector Search)"))
    logs.append(("DEBUG", f"  Query        : {query}"))
    logs.append(("DEBUG", f"  Chroma host  : {CHROMA_HOST}:{CHROMA_PORT}"))
    logs.append(("DEBUG", f"  Collection   : {COLLECTION_NAME}"))
    logs.append(("DEBUG",  "  Embedding model: all-MiniLM-L6-v2 (384 dimensions)"))

    try:
        model = _get_embed_model()
        t0 = time.time()
        embedding = model.encode([query])
        logs.append(("DEBUG", f"  Encoded in   : {time.time()-t0:.3f}s"))
        logs.append(("DEBUG", f"  Vector shape : {embedding.shape}  (1 x {embedding.shape[1]} dims)"))
        logs.append(("DEBUG", f"  Vector sample: [{embedding[0][0]:.4f}, {embedding[0][1]:.4f}, "
                               f"{embedding[0][2]:.4f} ... {embedding[0][-1]:.4f}]"))

        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        col = client.get_collection(COLLECTION_NAME)
        logs.append(("DEBUG", f"  Collection count: {col.count()} chunks stored"))

        t1 = time.time()
        results = col.query(query_embeddings=embedding.tolist(), n_results=3)
        elapsed = time.time() - t1
        logs.append(("DEBUG", f"  Search time  : {elapsed:.3f}s"))

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        logs.append(("OK",   f"  Chunks returned: {len(docs)}"))
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            preview = doc[:80].replace("\n", " ")
            logs.append(("OK",   f"  Chunk {i}: source={meta.get('source','?')}  "
                                  f"dist={dist:.4f}  preview=\"{preview}...\""))

        return "\n\n---\n\n".join(docs) if docs else "No relevant policy found."

    except Exception as e:
        logs.append(("ERROR", f"  ChromaDB error: {e}"))
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
    """Kept for UI compatibility."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            logger.warning("Model '%s' not found. Available: %s", OLLAMA_MODEL, models)
    except Exception as e:
        logger.warning("Could not reach Ollama at %s: %s", OLLAMA_BASE_URL, e)
    return None


if __name__ == "__main__":
    print("HR AI Assistant - Debug Pipeline Test")
    print("=" * 55)
    q = "How many holiday days do I have left? My ID is EMP001"
    answer, logs = ask(q)
    for level, msg in logs:
        print(f"[{level:<5}] {msg}")
    print(f"\nANSWER:\n{answer}")
