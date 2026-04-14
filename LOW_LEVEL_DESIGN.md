# AI-Powered HR Assistant — Low Level Design Document

**Version:** 1.0  
**Date:** April 2026  
**Stack:** Python 3.13 · PostgreSQL 16 · ChromaDB · Ollama 0.20.6 · Llama 3 · Streamlit 1.56  

---

## Table of Contents

1. [System Purpose](#1-system-purpose)
2. [High Level Architecture](#2-high-level-architecture)
3. [Component Design](#3-component-design)
   - 3.1 [PostgreSQL Database](#31-postgresql-database)
   - 3.2 [ChromaDB Vector Store](#32-chromadb-vector-store)
   - 3.3 [PDF Ingestion Pipeline](#33-pdf-ingestion-pipeline)
   - 3.4 [MCP Server](#34-mcp-server)
   - 3.5 [RAG Orchestrator](#35-rag-orchestrator)
   - 3.6 [Streamlit Chat UI](#36-streamlit-chat-ui)
4. [Data Models](#4-data-models)
5. [API Contracts](#5-api-contracts)
6. [Request Flow — Step by Step](#6-request-flow--step-by-step)
7. [Environment Configuration](#7-environment-configuration)
8. [Docker Infrastructure](#8-docker-infrastructure)
9. [Issues Encountered and Fixes Applied](#9-issues-encountered-and-fixes-applied)
10. [Design Decisions and Trade-offs](#10-design-decisions-and-trade-offs)
11. [Known Limitations](#11-known-limitations)
12. [Future Improvements](#12-future-improvements)

---

## 1. System Purpose

### Problem Statement

Employees frequently contact HR with routine questions such as:
- "How many holiday days do I have left?"
- "What is the sick leave policy?"
- "Can I carry unused leave to next year?"

These questions require cross-referencing two data sources:
1. The HR database (how many days has this employee actually taken?)
2. Company policy documents (what is their total entitlement?)

Currently this involves email threads and manual HR intervention — slow and inefficient.

### Solution

An AI assistant that:
- Accepts natural language questions
- Automatically queries both data sources
- Returns a personalised, accurate answer in seconds
- Runs 100% locally with no external API costs

---

## 2. High Level Architecture

```
+------------------+
|   Employee       |
|   (Browser)      |
+--------+---------+
         |
         | HTTP (port 8501)
         v
+------------------+
|  Streamlit UI    |  ui/app.py
|  (Chat Interface)|
+--------+---------+
         |
         | Python function call
         v
+------------------+
|  RAG Orchestrator|  rag/orchestrator.py
|  (Pipeline)      |
+--------+---------+
         |
    +----+----+
    |         |
    v         v
+-------+  +----------+
|  MCP  |  |   MCP    |
| Tool 1|  |  Tool 2  |
|  DB   |  |  Policy  |
+---+---+  +----+-----+
    |            |
    v            v
+----------+ +----------+
|PostgreSQL| | ChromaDB |
|(port 5432| |(port 8000|
+----------+ +----------+
                  ^
                  |
         +--------+--------+
         | Ingestion Pipeline|
         | ingest_pdfs.py   |
         +--------+---------+
                  |
         +--------+--------+
         | HuggingFace      |
         | Policy PDFs      |
         +------------------+

All LLM calls:
RAG Orchestrator --> Ollama REST API (port 11434) --> Llama 3 (local)
```

---

## 3. Component Design

### 3.1 PostgreSQL Database

**Container:** `hr_postgres` (postgres:16-alpine)  
**Port:** 5432  
**Database:** hrdb  
**Credentials:** hruser / hrpassword  

#### Tables

**employees**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | SERIAL | PRIMARY KEY | Auto-increment row ID |
| employee_id | VARCHAR(20) | NOT NULL, UNIQUE | Business key e.g. EMP001 |
| name | VARCHAR(100) | NOT NULL | Full name |
| email | VARCHAR(150) | | Work email |
| department | VARCHAR(100) | | Department name |
| employment_type | VARCHAR(20) | DEFAULT 'full-time' | full-time or part-time |

**employee_leave**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | SERIAL | PRIMARY KEY | Auto-increment row ID |
| employee_id | VARCHAR(20) | FK -> employees | Links to employee |
| year | INT | NOT NULL | Calendar year |
| days_taken | INT | NOT NULL, DEFAULT 0 | Days used so far |
| days_pending | INT | NOT NULL, DEFAULT 0 | Approved but not yet taken |

**Unique constraint:** `(employee_id, year)` — one record per employee per year.

#### Seed Data

```sql
-- Employees
('EMP001', 'John Smith',   'Engineering',  'full-time')
('EMP002', 'Sara Jones',   'Marketing',    'full-time')
('EMP003', 'Ahmed Hassan', 'Finance',       'full-time')
('EMP004', 'Lisa Chen',    'HR',           'part-time')

-- Leave records (current year)
('EMP001', days_taken=17, days_pending=2)
('EMP002', days_taken=8,  days_pending=0)
('EMP003', days_taken=22, days_pending=0)
('EMP004', days_taken=5,  days_pending=1)
```

#### Key SQL Query (used in get_leave_info)

```sql
SELECT e.name, e.department, e.employment_type,
       l.year, l.days_taken, l.days_pending
FROM employees e
LEFT JOIN employee_leave l
    ON e.employee_id = l.employee_id
    AND l.year = EXTRACT(YEAR FROM CURRENT_DATE)
WHERE e.employee_id = %s
```

Uses LEFT JOIN so employees without leave records still return data.

---

### 3.2 ChromaDB Vector Store

**Container:** `hr_chromadb` (chromadb/chroma:latest)  
**Port:** 8000  
**Collection:** `hr_policy`  
**Similarity metric:** Cosine  
**Persistence:** Docker volume `chroma_data`

#### Document Schema (per chunk)

| Field | Type | Example |
|-------|------|---------|
| id | string | `annual_leave_policy_0` |
| document | string | Raw text chunk (500 chars) |
| embedding | float[] | 384-dimension vector |
| metadata.source | string | `annual_leave_policy.pdf` |
| metadata.chunk | int | `0` |

#### Embedding Model

Model: `sentence-transformers/all-MiniLM-L6-v2`  
Dimensions: 384  
Runtime: Local CPU (no GPU required)  
Size: ~90 MB  

Chosen because:
- Free, no API key needed
- Fast on CPU
- High quality for English semantic search
- Works offline

#### Query Process

```python
query_embedding = model.encode([query])       # 384-dim vector
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3                                # top 3 chunks
)
# Returns most semantically similar policy text
```

---

### 3.3 PDF Ingestion Pipeline

#### Stage A — Download (download_policies.py)

```
HuggingFace dataset (syncora/hr-policies-qa-dataset)
    |
    | datasets.load_dataset()
    v
644 records with 'messages' field
    |
    | keyword classification
    v
6 topic buckets
    |
    | fpdf2 PDF generation
    v
6 PDF files in ./policies/
```

**Topic classification logic:**

```python
TOPIC_MAP = {
    "annual_leave_policy":        ["annual leave", "holiday", "vacation", "pto"],
    "sick_leave_policy":          ["sick leave", "sick day", "medical leave"],
    "maternity_paternity_policy": ["maternity", "paternity", "parental leave"],
    "remote_work_policy":         ["remote work", "work from home", "wfh"],
    "performance_policy":         ["performance review", "appraisal", "kpi"],
    "recruitment_policy":         ["recruitment", "hiring", "interview"],
    "general_hr_policy":          [],   # catch-all
}
```

Each record's text is lowercased and scanned for keywords.
First match wins. Unmatched records go to `general_hr_policy`.

#### Stage B — Ingest (ingest_pdfs.py)

```
PDF file
    |
    | pdfplumber.open().pages[n].extract_text()
    v
Raw text string
    |
    | chunk_text(size=500, overlap=50)
    v
List of text chunks
    |
    | SentenceTransformer.encode(chunks)
    v
List of 384-dim embeddings
    |
    | ChromaDB collection.add()
    v
Stored in vector store
```

**Chunking parameters:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| Chunk size | 500 chars | Fits within LLM context window comfortably |
| Overlap | 50 chars | Prevents cutting sentences at boundaries |
| Model | all-MiniLM-L6-v2 | Best speed/quality for local CPU |

**Total ingested:** 92 chunks from 6 PDFs

---

### 3.4 MCP Server

**File:** `mcp_server/server.py`  
**Framework:** FastMCP  
**Pattern:** Model Context Protocol — exposes Python functions as callable tools

#### Tool 1: get_leave_info

```
Input:  employee_id (str)  e.g. "EMP001"
Output: dict {
    employee_id:      "EMP001"
    name:             "John Smith"
    department:       "Engineering"
    employment_type:  "full-time"
    year:             2026
    days_taken:       17
    days_pending:     2
}
Error:  dict { "error": "No employee found with ID EMP999" }
```

**Implementation:**
- Opens psycopg2 connection per call (simple, no connection pooling needed at this scale)
- Returns error dict rather than raising exception — safe for LLM consumption
- Normalises employee_id to uppercase before query

#### Tool 2: search_policy

```
Input:  query (str)  e.g. "annual leave entitlement full-time"
Output: str (top 3 matching chunks joined by "---")
Error:  str "Policy search error: <message>"
```

**Implementation:**
- Encodes query using same sentence-transformers model as ingestion
- Queries ChromaDB with cosine similarity
- Returns top 3 document chunks concatenated

#### Tool 3: list_employees (admin)

```
Input:  none
Output: list of dicts [ { employee_id, name, department }, ... ]
```

Used by Streamlit sidebar to display the employee directory.

---

### 3.5 RAG Orchestrator

**File:** `rag/orchestrator.py`  
**Pattern:** Fetch-then-Prompt (no agent loop, no tool-calling API)

#### Why this pattern?

Native tool-calling (LangGraph ReAct agent) was the original design but was abandoned because:
- `llama3` base model does not support Ollama's tool-calling API (HTTP 400)
- Adds multi-turn overhead (slower)
- Harder to debug

The fetch-then-prompt pattern:
- Works with any Ollama model
- Single LLM call (faster)
- Deterministic data fetching
- Easier to understand and modify

#### Step-by-step logic

```python
def ask(question: str) -> str:

    # Step 1: Extract employee ID from question text
    # Regex: looks for pattern EMP followed by 3+ digits
    emp_id = re.search(r'\b(EMP\d{3,})\b', question.upper())

    # Step 2: Fetch structured leave data (if ID found)
    leave_data = get_leave_info(emp_id) if emp_id else None

    # Step 3: Semantic search for relevant policy text
    policy_text = search_policy(question)

    # Step 4: Build structured prompt
    prompt = build_prompt(question, leave_data, policy_text)

    # Step 5: Send to Ollama REST API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
        timeout=120
    )
    return response.json()["response"]
```

#### Prompt template

```
You are a friendly and accurate HR assistant.
Answer the employee's question using ONLY the data provided below.
Be warm, specific, and show the calculation clearly.

EMPLOYEE LEAVE RECORD:
  Name: {name}
  Employee ID: {employee_id}
  Department: {department}
  Employment type: {employment_type}
  Year: {year}
  Days taken so far: {days_taken}
  Days pending approval: {days_pending}

RELEVANT HR POLICY:
{policy_text}

EMPLOYEE QUESTION: {question}

YOUR ANSWER:
```

The prompt instructs the LLM to:
1. Only use provided data (no hallucination)
2. Be warm and friendly in tone
3. Show the calculation (entitlement - taken = remaining)

#### Ollama REST API

**Endpoint:** `POST /api/generate`  
**Payload:**
```json
{
  "model": "llama3",
  "prompt": "<full prompt string>",
  "stream": false
}
```
**Response:**
```json
{
  "response": "Hi John! You have 6 days remaining...",
  "done": true
}
```

`stream: false` waits for the full response before returning. Simpler than streaming for this use case.

---

### 3.6 Streamlit Chat UI

**File:** `ui/app.py`  
**Port:** 8501  
**Framework:** Streamlit 1.56

#### Session State

| Key | Type | Initial Value | Purpose |
|-----|------|--------------|---------|
| `messages` | list | Welcome message | Stores full chat history |
| `agent_ready` | bool | Set on first load | Is Ollama reachable? |
| `agent_error` | str | "" | Error detail if not ready |

#### Page Layout

```
+----------------------------------+------------------+
|  HR AI Assistant                 |   Sidebar        |
|                                  |                  |
|  [Welcome message]               | Employee Dir:    |
|                                  | EMP001 - John    |
|  [User message]                  | EMP002 - Sara    |
|                                  | EMP003 - Ahmed   |
|  [Assistant reply]               | EMP004 - Lisa    |
|                                  |                  |
|  [User message]                  | System Status:   |
|                                  | [Ollama ready]   |
|  [Assistant reply]               |                  |
|                                  | [Clear Chat btn] |
|  +--------------------------+    |                  |
|  | Ask your question here...|    |                  |
|  +--------------------------+    |                  |
+----------------------------------+------------------+
```

#### Message flow

```
User types message
    |
    v
Append to st.session_state.messages
    |
    v
Display user message in chat
    |
    v
Show "Thinking..." spinner
    |
    v
Call orchestrator.ask(prompt)
    |
    v
Display assistant response
    |
    v
Append response to st.session_state.messages
```

---

## 4. Data Models

### Employee Leave Response (from get_leave_info)

```python
{
    "employee_id":      str,    # "EMP001"
    "name":             str,    # "John Smith"
    "department":       str,    # "Engineering"
    "employment_type":  str,    # "full-time" | "part-time"
    "year":             int,    # 2026
    "days_taken":       int,    # 17
    "days_pending":     int,    # 2
}
```

### ChromaDB Query Result

```python
{
    "documents": [["chunk text 1", "chunk text 2", "chunk text 3"]],
    "metadatas": [[{"source": "annual_leave_policy.pdf", "chunk": 0}, ...]],
    "distances": [[0.12, 0.23, 0.31]]
}
```

### Ollama API Request

```python
{
    "model":  str,   # "llama3"
    "prompt": str,   # full assembled prompt
    "stream": bool   # False for synchronous response
}
```

### Ollama API Response

```python
{
    "model":       str,   # "llama3"
    "response":    str,   # generated answer text
    "done":        bool,  # True when complete
    "total_duration": int # nanoseconds
}
```

---

## 5. API Contracts

### Internal Python APIs

#### orchestrator.ask(question, agent=None) -> str
- **Input:** Natural language question string
- **Output:** Natural language answer string
- **Side effects:** Queries PostgreSQL and ChromaDB
- **Timeout:** 120 seconds (Ollama generation)
- **Error:** Returns error string (never raises)

#### mcp_server.get_leave_info(employee_id) -> dict
- **Input:** Employee ID string (case-insensitive)
- **Output:** Leave data dict or `{"error": "..."}` dict
- **Side effects:** Opens and closes DB connection
- **Error handling:** Returns error dict on exception

#### mcp_server.search_policy(query) -> str
- **Input:** Plain English query string
- **Output:** Concatenated policy text chunks
- **Side effects:** Calls ChromaDB HTTP API
- **Error handling:** Returns error string on exception

### External HTTP APIs

#### Ollama: POST /api/generate
- **Base URL:** http://localhost:11434
- **Auth:** None (local only)
- **Timeout:** 120s

#### ChromaDB: HTTP REST
- **Base URL:** http://localhost:8000
- **Auth:** None (local only)
- **Used via:** chromadb.HttpClient Python SDK

#### PostgreSQL
- **Protocol:** TCP port 5432
- **Auth:** username/password
- **Used via:** psycopg2 Python driver

---

## 6. Request Flow — Step by Step

**Scenario:** John Smith asks "How many holidays do I have left? EMP001"

```
Step 1:  User types message in Streamlit text input
         Message: "How many holidays do I have left? EMP001"

Step 2:  Streamlit appends to session_state.messages
         Displays user message bubble in chat

Step 3:  orchestrator.ask("How many holidays do I have left? EMP001") called

Step 4:  Regex search on question.upper()
         Pattern: r'\b(EMP\d{3,})\b'
         Match found: "EMP001"

Step 5:  get_leave_info("EMP001") called
         Opens psycopg2 connection to PostgreSQL
         Executes JOIN query
         Returns: {name: "John Smith", days_taken: 17, days_pending: 2, ...}
         Connection closed

Step 6:  search_policy("How many holidays do I have left? EMP001") called
         Encodes query -> 384-dim vector
         ChromaDB cosine search -> top 3 chunks from hr_policy collection
         Returns: "Full-time employees are entitled to 25 days..."

Step 7:  _build_prompt() assembles:
         [System instruction]
         [Employee leave record block]
         [Policy text block]
         [Original question]
         [YOUR ANSWER: prompt]

Step 8:  requests.post("http://localhost:11434/api/generate", ...)
         Ollama receives prompt
         Llama 3 generates response (~5-15 seconds)
         Returns: "Hi John! You have 6 days remaining. Your entitlement
                   is 25 days and you have taken 17 so far this year..."

Step 9:  Response string returned to Streamlit

Step 10: Streamlit displays response in assistant chat bubble
         Appends to session_state.messages
```

---

## 7. Environment Configuration

**File:** `.env` (copied from `.env.example`, never committed to git)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgresql://hruser:hrpassword@localhost:5432/hrdb | PostgreSQL connection string |
| `CHROMA_HOST` | localhost | ChromaDB host |
| `CHROMA_PORT` | 8000 | ChromaDB port |
| `CHROMA_COLLECTION` | hr_policy | Vector store collection name |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API base URL |
| `OLLAMA_MODEL` | llama3 | Model name to use |
| `POLICIES_DIR` | ./policies | Directory containing PDF files |

---

## 8. Docker Infrastructure

**File:** `docker-compose.yml`

### Services

#### postgres
- **Image:** postgres:16-alpine (minimal image, ~80MB)
- **Init:** `./db/init.sql` auto-runs on first start via `docker-entrypoint-initdb.d`
- **Persistence:** Named volume `postgres_data`
- **Health check:** `pg_isready -U hruser -d hrdb` every 5 seconds

#### chromadb
- **Image:** chromadb/chroma:latest
- **Persistence:** Named volume `chroma_data`
- **Persistence env:** `IS_PERSISTENT=TRUE`

### Volumes

| Volume | Mounted in | Purpose |
|--------|-----------|---------|
| postgres_data | /var/lib/postgresql/data | Survives container restarts |
| chroma_data | /chroma/chroma | Survives container restarts |

### Commands

```bash
# Start both services in background
docker-compose up -d

# Check status
docker ps

# View logs
docker logs hr_postgres
docker logs hr_chromadb

# Stop (keeps data)
docker-compose down

# Stop and delete all data
docker-compose down -v
```

---

## 9. Issues Encountered and Fixes Applied

---

### Issue 1 — Docker Daemon Not Running

**Stage:** Initial docker-compose up  
**Error:**
```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine;
check if the path is correct and if the daemon is running
```
**Root Cause:** Docker Desktop was not started before running docker-compose.  
**Fix:** Start Docker Desktop from Windows Start menu and wait ~30 seconds for it to fully initialise before running docker-compose commands.

---

### Issue 2 — Python Not Found in Bash PATH

**Stage:** Running pip install  
**Error:**
```
pip: command not found
python: command not found
```
**Root Cause:** Python was installed via Microsoft Store. The Store version places the executable at:
`C:\Users\vdeva\AppData\Local\Microsoft\WindowsApps\python.exe`
This path is not on the bash PATH inside Claude Code's shell environment.

**Fix:** Use the full absolute path explicitly:
```bash
powershell.exe -Command "& 'C:\Users\vdeva\AppData\Local\Microsoft\WindowsApps\python.exe' -m pip install ..."
```
Or use PowerShell directly for all Python commands.

---

### Issue 3 — fpdf2 Unicode Encoding Error

**Stage:** download_policies.py — PDF generation  
**Error:**
```
fpdf.errors.FPDFUnicodeEncodingException: Character "—" at index 27
in text is outside the range of characters supported by the font
used: "helveticaI". Please consider using a Unicode font.
```
**Root Cause:** The HuggingFace dataset contains Unicode characters (em dash `\u2014`, smart quotes `\u201c`, `\u201d`) that are outside the Latin-1 range supported by fpdf2's built-in Helvetica font.

**Fix:** Added `sanitize()` function applied to all text before writing to PDF:
```python
def sanitize(text: str) -> str:
    replacements = {
        "\u2014": "-",   # em dash -> hyphen
        "\u2013": "-",   # en dash -> hyphen
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00b7": "*",   # middle dot
        "\u2022": "*",   # bullet
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="ignore").decode("latin-1")
```

---

### Issue 4 — fpdf2 Deprecated `ln=True` Parameter

**Stage:** download_policies.py — PDF generation  
**Error:**
```
DeprecationWarning: The parameter "ln" is deprecated since fpdf2 v2.5.2.
Instead of ln=True use new_x=XPos.LMARGIN, new_y=YPos.NEXT.
```
**Root Cause:** fpdf2 version 2.5.2+ changed the cell() API. The old `ln=True` parameter was removed.

**Fix:** Updated all `cell()` calls:
```python
# Before (broken in fpdf2 >= 2.5.2)
pdf.cell(0, 12, title, ln=True, align="C")

# After (correct)
pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
```

---

### Issue 5 — Windows cp1252 Terminal Encoding Error

**Stage:** ingest_pdfs.py — console output  
**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
in position 2: character maps to <undefined>
```
**Root Cause:** Windows default terminal encoding is `cp1252` (Windows-1252), which does not include the `→` right arrow character (`\u2192`) used in a print statement.

**Fix:** Replaced the Unicode arrow with plain ASCII:
```python
# Before
print(f"  → {len(chunks)} chunks ingested")

# After
print(f"  >> {len(chunks)} chunks ingested")
```

---

### Issue 6 — LangChain v1.x Breaking Import Changes

**Stage:** rag/orchestrator.py — startup  
**Error:**
```
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```
**Root Cause:** LangChain v1.x (installed: 1.2.15) restructured the agents module. `AgentExecutor` and `create_react_agent` were removed from `langchain.agents` and moved to `langgraph`.

**Original broken imports:**
```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
```

**Attempted fix (LangGraph ReAct agent):**
```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
```
This resolved the import but led to Issue 7.

**Final fix:** Removed agents entirely. Replaced with a direct pipeline using `requests` to call the Ollama REST API directly. No LangChain agent framework required.

---

### Issue 7 — Llama 3 Does Not Support Tool-Calling API

**Stage:** RAG orchestrator — first chat message  
**Error:**
```
registry.ollama.ai/library/llama3:latest does not support tools
(status code: 400)
```
**Root Cause:** The `llama3` (8B base) model does not implement Ollama's native tool/function-calling interface. Only newer or specifically fine-tuned models support this:
- llama3.1 — supports tools
- llama3.2 — supports tools
- mistral — supports tools
- qwen2.5 — supports tools
- llama3 (base) — does NOT support tools

**Fix:** Redesigned the orchestrator to avoid tool-calling entirely:

```
Before (broken):
  LangGraph ReAct agent -> model.bind_tools([...]) -> Ollama tool API -> Error

After (working):
  Python directly calls get_leave_info() and search_policy()
  Results injected into prompt as plain text
  Single call to Ollama /api/generate with no tools parameter
```

This is architecturally simpler and actually faster (one LLM call vs multiple turns).

---

### Issue 8 — `sys` Used Before Import in orchestrator.py

**Stage:** rag/orchestrator.py — import  
**Error:** Would have caused `NameError: name 'sys' is not defined`  
**Root Cause:** A line `sys._called_from_orchestrator = True` was placed before the `import sys` statement during an early draft.

**Fix:** Reordered imports to place all standard library imports at the top of the file before any usage:
```python
import os
import sys   # <- moved to top
import json
import re
```

---

### Issue 9 — Streamlit Session State Missing `agent` Key

**Stage:** Streamlit UI — chat interaction  
**Error:**
```
st.session_state has no attribute "agent".
Did you forget to initialize it?
```
**Root Cause:** After the orchestrator was refactored to not use an agent object, the session state key `agent` was removed from initialization. However, line 101 in `ui/app.py` still referenced `st.session_state.agent` when calling `ask()`.

**Fix:** Updated the `ask()` call to not pass the agent:
```python
# Before
response = ask(prompt, st.session_state.agent)

# After
response = ask(prompt)
```
Also removed the `agent` key from session state initialization entirely since it was no longer needed.

---

### Issue 10 — HuggingFace Symlink Warning on Windows

**Stage:** download_policies.py and ingest_pdfs.py  
**Warning:**
```
huggingface_hub cache-system uses symlinks by default to efficiently
store duplicated files but your machine does not support them in
C:\Users\vdeva\.cache\huggingface\hub\...
Caching files will still work but in a degraded version that might
require more space on your disk.
```
**Root Cause:** Windows requires Developer Mode enabled to create symlinks without administrator privileges. The HuggingFace hub uses symlinks for deduplication in its cache.

**Impact:** No functional impact. Models and datasets download and work correctly. Slightly more disk space used.

**Fix options (optional):**
1. Enable Windows Developer Mode: Settings -> System -> Developer Mode
2. Suppress warning: add `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to `.env`

---

### Issue 11 — docker-compose `version` Attribute Warning

**Stage:** docker-compose up  
**Warning:**
```
the attribute `version` is obsolete, it will be ignored,
please remove it to avoid potential confusion
```
**Root Cause:** Docker Compose v2+ deprecated the top-level `version:` key in `docker-compose.yml`. The file contained `version: "3.9"`.

**Impact:** No functional impact. All services start correctly.

**Fix:** Remove the `version:` line from `docker-compose.yml`:
```yaml
# Remove this line:
version: "3.9"

services:
  postgres:
    ...
```

---

## 10. Design Decisions and Trade-offs

### Decision 1 — Direct pipeline vs ReAct agent

| Option | Pros | Cons |
|--------|------|------|
| ReAct agent (LangGraph) | Dynamic tool selection, extensible | Requires tool-calling model, slower, complex |
| Direct pipeline (chosen) | Works with any model, fast, simple | Less flexible for multi-step reasoning |

**Chosen:** Direct pipeline. For this use case (HR Q&A with two known data sources) the pipeline approach is sufficient and more reliable.

---

### Decision 2 — Local LLM (Ollama) vs Cloud API

| Option | Pros | Cons |
|--------|------|------|
| Ollama + Llama 3 (chosen) | Free, private, offline, no rate limits | Slower, requires 8GB+ RAM |
| OpenAI GPT-4 | Fast, high quality | Cost per token, data leaves machine |
| Anthropic Claude | High quality | Cost per token, data leaves machine |

**Chosen:** Ollama. Privacy and zero cost were the primary requirements.

---

### Decision 3 — ChromaDB vs pgvector

| Option | Pros | Cons |
|--------|------|------|
| ChromaDB (chosen) | Simple setup, dedicated vector store | Extra service to run |
| pgvector | Single DB for everything | Requires PostgreSQL extension |

**Chosen:** ChromaDB. Simpler mental model and purpose-built for vector search.

---

### Decision 4 — sentence-transformers vs OpenAI embeddings

| Option | Pros | Cons |
|--------|------|------|
| all-MiniLM-L6-v2 (chosen) | Free, local, fast | 384-dim (lower than 1536) |
| OpenAI text-embedding-3-small | 1536-dim, high quality | API cost, data leaves machine |

**Chosen:** all-MiniLM-L6-v2. Sufficient quality for HR policy retrieval, zero cost.

---

### Decision 5 — HuggingFace dataset vs real company PDFs

**Chosen:** HuggingFace dataset for demo purposes. The system is designed to accept real company PDFs — just drop them in `./policies/` and re-run `ingest_pdfs.py`.

---

## 11. Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| Employee ID must be in message | Cannot look up by name alone | Ask user to include their ID |
| llama3 base model — no tool-calling | Pipeline only (not agentic) | Switch to llama3.1 for tool-calling |
| No authentication | Anyone can view any employee data | Add login layer for production |
| Single year leave records | Cannot query historical years | Extend SQL query with year parameter |
| No leave request submission | Read-only assistant | Add write tools to MCP server |
| Part-time entitlement not auto-calculated | Policy text handles this | Extend DB schema with entitlement column |
| English language only | Policy search in English only | Add multilingual embedding model |

---

## 12. Future Improvements

### Short Term
- Add employee login / authentication
- Support querying by name (not just ID)
- Add leave request submission tool
- Add historical leave query (by year)
- Store entitlement in DB per employee type

### Medium Term
- Replace llama3 with llama3.1 for native tool-calling
- Add conversation memory (multi-turn context)
- Add admin dashboard for HR managers
- Email/Slack notification integration
- Multi-language support

### Long Term
- Deploy to cloud (AWS ECS / Azure Container Apps)
- Replace Streamlit with React frontend
- Add analytics dashboard
- Integrate with real HRIS (Workday, SAP SuccessFactors)
- Add document upload UI for new HR policies

---

*End of Low Level Design Document*  
*Last updated: April 2026*
