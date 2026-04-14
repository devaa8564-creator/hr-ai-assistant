# AI-Powered HR Assistant — Complete Build Guide

> 100% Open Source · Runs locally · No external API costs  
> Stack: Python · PostgreSQL · ChromaDB · Ollama (Llama 3) · LangChain · Streamlit

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Stage 1 — Project Scaffold](#4-stage-1--project-scaffold)
5. [Stage 2 — Database Setup (PostgreSQL)](#5-stage-2--database-setup-postgresql)
6. [Stage 3 — Policy PDFs from HuggingFace](#6-stage-3--policy-pdfs-from-huggingface)
7. [Stage 4 — Vector Store Ingestion (ChromaDB)](#7-stage-4--vector-store-ingestion-chromadb)
8. [Stage 5 — MCP Server](#8-stage-5--mcp-server)
9. [Stage 6 — RAG Orchestrator](#9-stage-6--rag-orchestrator)
10. [Stage 7 — Streamlit Chat UI](#10-stage-7--streamlit-chat-ui)
11. [Issues & Fixes Log](#11-issues--fixes-log)
12. [Quick-Start Cheatsheet](#12-quick-start-cheatsheet)

---

## 1. System Overview

### What it does
Employees ask HR questions in plain English and get instant, accurate answers:

```
Employee:  "How many holiday days do I have left? My ID is EMP001"
Assistant: "Hi John! You have 6 days remaining. Your entitlement is 25 days
            and you have taken 17 so far this year, with 2 pending approval."
```

### Two data sources combined

| Source | What it stores | Technology |
|--------|---------------|------------|
| HR Database | Days taken, employee records | PostgreSQL |
| Policy Documents | Leave entitlements, rules | ChromaDB (vector store) |

### How they work together

```
Employee question
      |
      v
RAG Orchestrator (Python)
   |            |
   v            v
PostgreSQL    ChromaDB
(days taken)  (policy text)
   |            |
   +-----+------+
         |
         v
   Ollama / Llama 3
   (generates answer)
         |
         v
   Streamlit UI
```

---

## 2. Architecture

### Full component map

```
hr-ai-assistant/
├── docker-compose.yml       # PostgreSQL + ChromaDB containers
├── requirements.txt         # All Python dependencies
├── .env                     # Config (DB URL, Ollama URL, model name)
│
├── db/
│   └── init.sql             # Schema + seed data (runs on first container start)
│
├── ingest/
│   ├── download_policies.py # Downloads HR Q&A dataset from HuggingFace -> PDFs
│   └── ingest_pdfs.py       # Reads PDFs, chunks text, embeds, stores in ChromaDB
│
├── mcp_server/
│   └── server.py            # Two tools: get_leave_info() + search_policy()
│
├── rag/
│   └── orchestrator.py      # Pipeline: fetch data -> build prompt -> call Ollama
│
├── ui/
│   └── app.py               # Streamlit web chat interface
│
└── policies/                # PDF files (generated or your own)
```

### Data flow (per question)

```
1. User types question in Streamlit UI
2. orchestrator.ask() is called
3. Employee ID extracted from text (regex: EMP\d+)
4. get_leave_info(emp_id) -> queries PostgreSQL -> returns days_taken, name, dept
5. search_policy(question) -> queries ChromaDB -> returns top 3 policy chunks
6. Prompt assembled: [system instruction + employee record + policy text + question]
7. Prompt sent to Ollama /api/generate (llama3, no tool-calling required)
8. Response displayed in chat UI
```

---

## 3. Prerequisites

| Tool | Version used | Purpose |
|------|-------------|---------|
| Windows 11 | 10.0.26200 | OS |
| Python | 3.13 (Microsoft Store) | Runtime |
| Docker Desktop | Latest | Run PostgreSQL + ChromaDB |
| Ollama | 0.20.6 | Run Llama 3 locally |
| Llama 3 | 4.7 GB | LLM model |

### Install order

```
1. Docker Desktop   -> docker.com/products/docker-desktop
2. Python 3.11+     -> python.org/downloads  (check "Add to PATH")
3. Ollama           -> ollama.com/download   (auto-starts as Windows service)
```

---

## 4. Stage 1 — Project Scaffold

### What was created

```
mkdir hr-ai-assistant
cd hr-ai-assistant
mkdir db ingest mcp_server rag ui policies
```

### Key files

**`.env.example`** — all configurable values:
```env
DATABASE_URL=postgresql://hruser:hrpassword@localhost:5432/hrdb
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=hr_policy
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
POLICIES_DIR=./policies
```

**`requirements.txt`**:
```
fastmcp>=0.4.0
psycopg2-binary>=2.9.9
chromadb>=0.5.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-ollama>=0.1.0
langchain-core
langgraph
sentence-transformers>=3.0.0
pymupdf>=1.24.0
pdfplumber>=0.11.0
streamlit>=1.35.0
python-dotenv>=1.0.0
datasets>=2.19.0
fpdf2>=2.7.9
requests
```

**Install all deps:**
```cmd
python -m pip install -r requirements.txt
```

---

## 5. Stage 2 — Database Setup (PostgreSQL)

### docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: hr_postgres
    environment:
      POSTGRES_DB: hrdb
      POSTGRES_USER: hruser
      POSTGRES_PASSWORD: hrpassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql

  chromadb:
    image: chromadb/chroma:latest
    container_name: hr_chromadb
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
```

### db/init.sql — Schema

```sql
CREATE TABLE IF NOT EXISTS employees (
    id              SERIAL PRIMARY KEY,
    employee_id     VARCHAR(20) NOT NULL UNIQUE,
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(150),
    department      VARCHAR(100),
    employment_type VARCHAR(20) DEFAULT 'full-time'
);

CREATE TABLE IF NOT EXISTS employee_leave (
    id            SERIAL PRIMARY KEY,
    employee_id   VARCHAR(20) NOT NULL REFERENCES employees(employee_id),
    year          INT NOT NULL DEFAULT EXTRACT(YEAR FROM CURRENT_DATE),
    days_taken    INT NOT NULL DEFAULT 0,
    days_pending  INT NOT NULL DEFAULT 0,
    UNIQUE (employee_id, year)
);
```

### Seed data (4 sample employees)

| ID | Name | Dept | Days Taken | Days Pending |
|----|------|------|-----------|--------------|
| EMP001 | John Smith | Engineering | 17 | 2 |
| EMP002 | Sara Jones | Marketing | 8 | 0 |
| EMP003 | Ahmed Hassan | Finance | 22 | 0 |
| EMP004 | Lisa Chen | HR (part-time) | 5 | 1 |

### Start containers

```cmd
docker-compose up -d
```

PostgreSQL auto-runs `init.sql` on first start. Takes ~10 seconds to be ready.

---

## 6. Stage 3 — Policy PDFs from HuggingFace

### Dataset used

**`syncora/hr-policies-qa-dataset`** on HuggingFace  
- 644 records of HR policy Q&A conversations  
- Fields: `messages` (list of system/user/assistant turns)

### Script: `ingest/download_policies.py`

**What it does:**
1. Loads dataset via `datasets` library
2. Classifies each record by topic using keyword matching
3. Groups records into topic buckets
4. Generates one PDF per topic using `fpdf2`

**Topic classification:**

| PDF File | Keywords matched | Records |
|----------|-----------------|---------|
| `annual_leave_policy.pdf` | holiday, vacation, leave balance | 3 |
| `sick_leave_policy.pdf` | sick leave, medical certificate | 3 |
| `remote_work_policy.pdf` | remote work, WFH, hybrid | 26 |
| `recruitment_policy.pdf` | hiring, interview, onboarding | 7 |
| `performance_policy.pdf` | appraisal, KPI, review | 4 |
| `general_hr_policy.pdf` | everything else | 601 |

### Run it

```cmd
cd hr-ai-assistant
python ingest\download_policies.py
```

**Output:** 6 PDF files in `./policies/`

> To use your own company PDFs: drop them in `./policies/` and re-run `ingest_pdfs.py`

---

## 7. Stage 4 — Vector Store Ingestion (ChromaDB)

### Script: `ingest/ingest_pdfs.py`

**Pipeline:**
```
PDF file
  -> pdfplumber extracts text
  -> split into 500-char chunks (50-char overlap)
  -> sentence-transformers encodes each chunk (all-MiniLM-L6-v2)
  -> ChromaDB stores (document, embedding, metadata)
```

**Key settings:**
```python
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
MODEL         = "all-MiniLM-L6-v2"  # free local embedding model
COLLECTION    = "hr_policy"
```

### Run it (after docker-compose up)

```cmd
python ingest\ingest_pdfs.py
```

**Output:**
```
Processing: annual_leave_policy.pdf   >>  3 chunks
Processing: general_hr_policy.pdf     >> 48 chunks
Processing: performance_policy.pdf    >>  4 chunks
Processing: recruitment_policy.pdf    >>  9 chunks
Processing: remote_work_policy.pdf    >> 24 chunks
Processing: sick_leave_policy.pdf     >>  4 chunks

Done! Total chunks stored: 92
```

---

## 8. Stage 5 — MCP Server

### File: `mcp_server/server.py`

Exposes two callable tools using FastMCP:

#### Tool 1: `get_leave_info(employee_id)`
```python
@mcp.tool()
def get_leave_info(employee_id: str) -> dict:
    # Queries PostgreSQL for employee + leave data
    # Returns: name, department, employment_type,
    #          year, days_taken, days_pending
```

#### Tool 2: `search_policy(query)`
```python
@mcp.tool()
def search_policy(query: str) -> str:
    # Embeds query with sentence-transformers
    # Queries ChromaDB for top 3 matching chunks
    # Returns: concatenated policy text
```

#### Tool 3: `list_employees()` (admin/debug)
```python
@mcp.tool()
def list_employees() -> list:
    # Returns all employees for sidebar display
```

### Run standalone (optional)

```cmd
python mcp_server\server.py
```

---

## 9. Stage 6 — RAG Orchestrator

### File: `rag/orchestrator.py`

**Design decision:** Uses a direct pipeline (not native tool-calling) so it works with any Ollama model including llama3.

### Pipeline steps

```python
def ask(question: str) -> str:
    # Step 1: Extract employee ID from free text
    emp_id = re.search(r'\b(EMP\d{3,})\b', question.upper())

    # Step 2: Fetch structured data
    leave_data = get_leave_info(emp_id)      # -> PostgreSQL

    # Step 3: Fetch policy context
    policy_text = search_policy(question)    # -> ChromaDB

    # Step 4: Build prompt
    prompt = f"""
    EMPLOYEE LEAVE RECORD:
      Name: {leave_data['name']}
      Days taken: {leave_data['days_taken']}
      Days pending: {leave_data['days_pending']}

    RELEVANT HR POLICY:
    {policy_text}

    EMPLOYEE QUESTION: {question}
    YOUR ANSWER:
    """

    # Step 5: Call Ollama REST API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]
```

### Ollama API call

```python
requests.post(
    f"{OLLAMA_BASE_URL}/api/generate",
    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    timeout=120,
)
```

No tool-calling, no agents — just a direct REST call. Works with any model.

---

## 10. Stage 7 — Streamlit Chat UI

### File: `ui/app.py`

**Features:**
- Chat history preserved in `st.session_state`
- Sidebar shows employee directory (live from PostgreSQL)
- Sidebar shows Ollama connection status
- Clear chat button
- Spinner while LLM is thinking

### Run it

```cmd
cd hr-ai-assistant
python -m streamlit run ui\app.py
```

**Opens at:** `http://localhost:8501`

### Session state keys

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | list | Chat history |
| `agent_ready` | bool | Ollama reachable? |
| `agent_error` | str | Error message if not ready |

---

## 11. Issues & Fixes Log

### Issue 1 — docker-compose `version` attribute warning
**Error:**
```
the attribute `version` is obsolete, it will be ignored
```
**Cause:** Newer Docker Compose versions deprecated the `version:` key.  
**Fix:** Safe to ignore — containers start correctly. Optionally remove `version: "3.9"` from `docker-compose.yml`.

---

### Issue 2 — fpdf2 Unicode encoding error
**Error:**
```
FPDFUnicodeEncodingException: Character "—" at index 27 is outside
the range of characters supported by the font used: "helveticaI"
```
**Cause:** Default Helvetica font in fpdf2 only supports Latin-1. The HuggingFace dataset contains Unicode characters (em dash `—`, smart quotes `""`).  
**Fix:** Added a `sanitize()` function that replaces common Unicode punctuation with ASCII equivalents before writing to PDF:
```python
def sanitize(text: str) -> str:
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="ignore").decode("latin-1")
```

---

### Issue 3 — fpdf2 deprecated `ln=True` parameter
**Error:**
```
DeprecationWarning: The parameter "ln" is deprecated since v2.5.2.
Instead of ln=True use new_x=XPos.LMARGIN, new_y=YPos.NEXT.
```
**Cause:** fpdf2 v2.5.2+ changed the cell() API.  
**Fix:** Replaced all `ln=True` with `new_x="LMARGIN", new_y="NEXT"`:
```python
# Before
pdf.cell(0, 12, title, ln=True, align="C")

# After
pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
```

---

### Issue 4 — Windows cp1252 encoding error in print statement
**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'
in position 2: character maps to <undefined>
```
**Cause:** Windows default terminal encoding is cp1252, which doesn't support the `→` arrow character used in a print statement.  
**Fix:** Replaced Unicode arrow with ASCII:
```python
# Before
print(f"  → {len(chunks)} chunks ingested")

# After
print(f"  >> {len(chunks)} chunks ingested")
```

---

### Issue 5 — LangChain v1.x import changes
**Error:**
```
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```
**Cause:** LangChain v1.x moved and restructured the agents module. `AgentExecutor` and `create_react_agent` are no longer available at the old import paths.  
**Fix:** Rewrote the orchestrator to not use LangChain agents at all. Instead uses a direct pipeline with raw `requests` to the Ollama REST API. Simpler, faster, and works with any model.

---

### Issue 6 — Llama 3 does not support tool-calling
**Error:**
```
registry.ollama.ai/library/llama3:latest does not support tools
(status code: 400)
```
**Cause:** The base `llama3` model does not implement Ollama's tool/function-calling API. Only newer models (`llama3.1`, `llama3.2`, `mistral`, `qwen2.5`) support this.  
**Fix:** Removed native tool-calling entirely. Rewrote orchestrator as a fetch-then-prompt pipeline:
1. Call `get_leave_info()` and `search_policy()` directly in Python
2. Inject results into the prompt as plain text context
3. Send to Ollama `/api/generate` (no tools parameter)

This approach works with **any** Ollama model and is actually faster than agent-based tool-calling.

---

### Issue 7 — `st.session_state has no attribute "agent"`
**Error:**
```
st.session_state has no attribute "agent".
Did you forget to initialize it?
```
**Cause:** After refactoring the orchestrator to not use an agent object, the UI still referenced `st.session_state.agent` on line 101.  
**Fix:** Removed the unused `agent` state variable and updated the `ask()` call:
```python
# Before
response = ask(prompt, st.session_state.agent)

# After
response = ask(prompt)
```

---

### Issue 8 — HuggingFace symlink warning on Windows
**Warning:**
```
huggingface_hub cache-system uses symlinks by default but your machine
does not support them. Caching will work in degraded mode.
```
**Cause:** Windows requires Developer Mode enabled to create symlinks without admin rights.  
**Impact:** None — caching still works, just uses more disk space.  
**Fix:** Safe to ignore. To suppress: set environment variable `HF_HUB_DISABLE_SYMLINKS_WARNING=1` in `.env`.

---

### Issue 9 — Python not found in bash PATH
**Error:**
```
python: command not found
pip: command not found
```
**Cause:** Python was installed via Microsoft Store, which places it in `AppData\Local\Microsoft\WindowsApps` — not on the bash PATH inside Claude Code's shell.  
**Fix:** Used the full path:
```
C:\Users\vdeva\AppData\Local\Microsoft\WindowsApps\python.exe
```
Or use PowerShell directly: `powershell.exe -Command "python ..."`

---

## 12. Quick-Start Cheatsheet

Use this every time you want to start the system from scratch.

### Step 1 — Start the databases
```cmd
cd C:\Users\vdeva\hr-ai-assistant
docker-compose up -d
```

### Step 2 — Ingest PDFs (first time only, or after adding new PDFs)
```cmd
python ingest\download_policies.py
python ingest\ingest_pdfs.py
```

### Step 3 — Launch the UI
```cmd
python -m streamlit run ui\app.py
```

Open browser: **http://localhost:8501**

### Step 4 — Stop everything
```cmd
docker-compose down
```
Ollama stops automatically when you close it from the system tray.

---

### Test questions to try

| Question | What it tests |
|----------|--------------|
| `How many days off do I have left? My ID is EMP001` | DB lookup + policy RAG |
| `What is the sick leave policy?` | Policy RAG only |
| `How many days has Ahmed Hassan taken? EMP003` | DB lookup only |
| `Can I carry over unused leave to next year?` | Policy RAG only |
| `What is the remote working policy?` | Policy RAG only |

---

### Environment variables (`.env`)

```env
DATABASE_URL=postgresql://hruser:hrpassword@localhost:5432/hrdb
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=hr_policy
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
POLICIES_DIR=./policies
```

> To use a different LLM: change `OLLAMA_MODEL` to `llama3.1`, `mistral`, `qwen2.5`, etc.  
> Then run: `ollama pull <model-name>`

---

*Generated: April 2026 | Stack: Python 3.13 · PostgreSQL 16 · ChromaDB · Ollama 0.20.6 · Llama 3 · Streamlit 1.56*
