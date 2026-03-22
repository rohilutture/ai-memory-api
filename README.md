# AI Memory API — FastAPI + MongoDB + Ollama

A study tutor chatbot with **three tiers of memory** built using FastAPI, MongoDB, and a local LLM via Ollama.

---

## What It Does

This project implements human-like memory for an AI tutor:

| Memory Tier | How It Works |
|---|---|
| **Short-term** | Sliding window of the last 6 messages passed directly into every prompt |
| **Long-term** | Session summaries generated every 5 messages; condensed into a lifetime student profile |
| **Episodic** | Up to 3 key facts extracted per message, embedded as vectors, retrieved by cosine similarity |

---

## Tech Stack

- **FastAPI** — async REST API
- **MongoDB** — persistent storage for all memory (reuses the `study_assistant` database)
- **Ollama** — local LLM inference (`phi3:mini` for chat, `nomic-embed-text` for embeddings)
- **httpx** — async HTTP client for Ollama calls

---

## MongoDB Collections

All collections are in the `study_assistant` database alongside Part 1's `tasks` collection (untouched).

| Collection | Fields |
|---|---|
| `messages` | `user_id`, `session_id`, `role`, `content`, `created_at` |
| `summaries` | `user_id`, `session_id`, `scope` (`session`\|`user`), `text`, `created_at` |
| `episodes` | `user_id`, `session_id`, `fact`, `importance` (0–1), `embedding` (float[768]), `created_at` |

---

## API Endpoints

### `POST /api/chat`
Send a message and get a reply with full memory context.
```json
{
  "user_id": "alice",
  "session_id": "1",
  "message": "I'm studying linked lists and struggling with pointers"
}
```
**Response:**
```json
{
  "reply": "...",
  "memory_used": {
    "short_term_count": 4,
    "short_term": [...],
    "session_summary": "...",
    "lifetime_summary": "...",
    "episodic_facts": [{"fact": "...", "importance": 0.95, "score": 0.88}]
  }
}
```

### `GET /api/memory/{user_id}`
Returns the full memory state for a user — last 16 messages, session summary, lifetime summary, and last 20 episodic facts.

### `GET /api/aggregate/{user_id}`
Returns daily message counts and recent summaries.

### `GET /api/health`
Health check confirming the chat and embed models in use.

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/rohilutture/ai-memory-api.git
cd ai-memory-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### 4. Configure environment
```bash
cp .env.example .env
```
Edit `.env` if needed — default values work out of the box with a local MongoDB instance.

### 5. Start MongoDB
```bash
# Using Docker (from Part 1)
docker-compose up -d

# Or install MongoDB locally and run as a Windows service
```

### 6. Run the API
```bash
uvicorn Main:app --reload --port 8000
```

Open **http://localhost:8000** for the chat UI, or **http://localhost:8000/docs** for the Swagger UI.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `study_assistant` | Database name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `CHAT_MODEL` | `phi3:mini` | Ollama model for chat |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `SHORT_TERM_N` | `6` | Number of messages in short-term window |
| `SUMMARIZE_EVERY_USER_MSGS` | `5` | Trigger summary every N user messages |
| `TOP_K_EPISODES` | `3` | Number of episodic facts retrieved per turn |

---

## Project Structure

```
ai-memory-api/
├── Main.py               # FastAPI app — all endpoints and memory logic
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # MongoDB via Docker (reused from Part 1)
├── .env.example          # Environment variable template
├── .gitignore            # Excludes .env and cache files
└── static/
    └── index.html        # Chat UI with Memory Inspector sidebar
```

---

## Memory Logic Overview

### Short-term memory
Fetches the last `SHORT_TERM_N` messages from MongoDB for the current session and injects them directly into the prompt.

### Long-term summaries
Every `SUMMARIZE_EVERY_USER_MSGS` user messages, a session summary is generated and stored. All session summaries are then condensed into a single lifetime student profile stored with `session_id: null`.

### Episodic memory
After every user message, a zero-temperature LLM call extracts up to 3 short facts with importance scores. Each fact is embedded using `nomic-embed-text` and stored in MongoDB. At query time, the current message is embedded and compared against all stored episodes using cosine similarity weighted by importance — the top-k facts are prepended to the prompt.

---
## Author

**Rohil Utture** 
