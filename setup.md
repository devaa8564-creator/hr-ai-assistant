# HR AI Assistant — Setup Guide

## Prerequisites

| Tool | Install |
|------|---------|
| Docker Desktop | https://www.docker.com/products/docker-desktop |
| Python 3.11+ | https://www.python.org/downloads |
| Ollama | https://ollama.com/download |

---

## Step-by-Step Setup

### 1. Install Python dependencies

```bash
cd hr-ai-assistant
pip install -r requirements.txt
```

### 2. Copy environment file

```bash
cp .env.example .env
```
Edit `.env` if your ports differ.

### 3. Start PostgreSQL + ChromaDB

```bash
docker-compose up -d
```

Wait ~10 seconds for both services to be ready.

### 4. Pull the Ollama model

```bash
ollama pull llama3
```

(First pull is ~4 GB — only needed once.)

### 5. Ingest policy PDFs into ChromaDB

Put your PDF files in `./policies/` (or skip — a sample policy is auto-loaded).

```bash
python ingest/ingest_pdfs.py
```

### 6. Start the Streamlit UI

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Test it

Try these questions in the chat:
- "How many holiday days do I have left? My ID is EMP001"
- "What is the sick leave policy?"
- "How much leave has Sara Jones (EMP002) taken?"

---

## Project Structure

```
hr-ai-assistant/
├── docker-compose.yml      # PostgreSQL + ChromaDB containers
├── requirements.txt
├── .env.example
├── db/
│   └── init.sql            # Schema + sample employee data
├── ingest/
│   └── ingest_pdfs.py      # PDF → ChromaDB ingestion
├── mcp_server/
│   └── server.py           # MCP tools: get_leave_info + search_policy
├── rag/
│   └── orchestrator.py     # LangChain agent (RAG + MCP)
├── ui/
│   └── app.py              # Streamlit chat interface
└── policies/               # Drop your HR policy PDFs here
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker ps` fails | Start Docker Desktop |
| Ollama not found | Run `ollama serve` in a separate terminal |
| ChromaDB connection refused | Run `docker-compose up -d` |
| Model slow to respond | Normal on first run — model loads into memory |
