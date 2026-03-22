"""
Part 2 — AI Memory FastAPI + MongoDB + Ollama
Short-term, long-term (summary), and episodic memory
"""

import os
import math
import json
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

MONGO_URI        = os.getenv("MONGO_URI",        "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGO_DB_NAME",    "study_assistant")   # same DB as Part 1
CHAT_MODEL       = os.getenv("CHAT_MODEL",       "phi3:mini")
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "nomic-embed-text")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
SHORT_TERM_N     = int(os.getenv("SHORT_TERM_N", "6"))
SUMMARIZE_EVERY  = int(os.getenv("SUMMARIZE_EVERY_USER_MSGS", "5"))
TOP_K_EPISODES   = int(os.getenv("TOP_K_EPISODES", "3"))

# ── MongoDB ───────────────────────────────────────────────────────────────────
mongo_client = MongoClient(MONGO_URI)
db           = mongo_client[DB_NAME]

messages_col  = db["messages"]   # NEW collection
summaries_col = db["summaries"]  # NEW collection
episodes_col  = db["episodes"]   # NEW collection

# Indexes for fast lookups
messages_col.create_index([("user_id", 1), ("session_id", 1), ("created_at", DESCENDING)])
summaries_col.create_index([("user_id", 1), ("scope", 1), ("created_at", DESCENDING)])
episodes_col.create_index([("user_id", 1), ("created_at", DESCENDING)])

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Memory API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id:    str
    session_id: Optional[str] = None
    message:    str

# ═════════════════════════════════════════════════════════════════════════════
#  OLLAMA HELPERS
# ═════════════════════════════════════════════════════════════════════════════

async def ollama_chat(messages: list[dict], model: str = CHAT_MODEL) -> str:
    """Call Ollama /api/chat and return assistant content string."""
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


async def ollama_embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Call Ollama /api/embeddings and return vector."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        r.raise_for_status()
        return r.json()["embedding"]


# ═════════════════════════════════════════════════════════════════════════════
#  MEMORY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Short-term memory ─────────────────────────────────────────────────────────

def get_short_term(user_id: str, session_id: str, n: int = SHORT_TERM_N) -> list[dict]:
    """Return last N messages for the session, oldest-first."""
    docs = list(
        messages_col.find(
            {"user_id": user_id, "session_id": session_id},
            {"_id": 0, "role": 1, "content": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(n)
    )
    return list(reversed(docs))


def save_message(user_id: str, session_id: str, role: str, content: str):
    messages_col.insert_one({
        "user_id":    user_id,
        "session_id": session_id,
        "role":       role,
        "content":    content,
        "created_at": now_iso(),
    })


# ── Long-term summaries ───────────────────────────────────────────────────────

def get_latest_summary(user_id: str, scope: str, session_id: Optional[str] = None) -> Optional[str]:
    q = {"user_id": user_id, "scope": scope}
    if scope == "session" and session_id:
        q["session_id"] = session_id
    doc = summaries_col.find_one(q, sort=[("created_at", DESCENDING)])
    return doc["text"] if doc else None


def count_user_messages(user_id: str, session_id: str) -> int:
    return messages_col.count_documents(
        {"user_id": user_id, "session_id": session_id, "role": "user"}
    )


async def maybe_summarize(user_id: str, session_id: str):
    """Trigger session summary every SUMMARIZE_EVERY user messages."""
    count = count_user_messages(user_id, session_id)
    if count % SUMMARIZE_EVERY != 0:
        return

    # Grab recent messages for summarization
    recent = list(
        messages_col.find(
            {"user_id": user_id, "session_id": session_id},
            {"_id": 0, "role": 1, "content": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(20)
    )
    recent = list(reversed(recent))

    convo_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    summary_prompt = [
        {"role": "system", "content": "You are a summarization assistant. Produce concise bullet-point summaries."},
        {"role": "user",   "content": (
            f"Summarize this tutoring conversation into 3-5 bullets covering topics discussed, "
            f"questions asked, and progress made:\n\n{convo_text}"
        )},
    ]

    try:
        summary_text = await ollama_chat(summary_prompt)
    except Exception as e:
        print(f"[summary error] {e}")
        return

    # Upsert session summary
    summaries_col.update_one(
        {"user_id": user_id, "session_id": session_id, "scope": "session"},
        {"$set": {"text": summary_text, "created_at": now_iso()}},
        upsert=True,
    )
    print(f"[summary] session summary updated for {user_id}/{session_id}")

    # Refresh lifetime summary from all session summaries
    all_session_summaries = list(
        summaries_col.find(
            {"user_id": user_id, "scope": "session"},
            {"_id": 0, "text": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(10)
    )
    if all_session_summaries:
        combined = "\n---\n".join(s["text"] for s in all_session_summaries)
        lifetime_prompt = [
            {"role": "system", "content": "You synthesize learning histories into student profiles."},
            {"role": "user",   "content": (
                f"Create a concise student profile (5-7 bullets) from these session summaries:\n\n{combined}"
            )},
        ]
        try:
            lifetime_text = await ollama_chat(lifetime_prompt)
            summaries_col.update_one(
                {"user_id": user_id, "scope": "user", "session_id": None},
                {"$set": {"text": lifetime_text, "created_at": now_iso(), "session_id": None}},
                upsert=True,
            )
            print(f"[summary] lifetime summary updated for {user_id}")
        except Exception as e:
            print(f"[lifetime summary error] {e}")


# ── Episodic memory ───────────────────────────────────────────────────────────

async def extract_and_store_episodes(user_id: str, session_id: str, user_msg: str):
    """Extract up to 3 facts from the user message, embed, and store."""
    extract_prompt = [
        {"role": "system", "content": (
            "You extract memorable facts from student messages. "
            "Return ONLY a JSON array of up to 3 objects with keys 'fact' (string) and 'importance' (0.0-1.0). "
            "No markdown, no explanation."
        )},
        {"role": "user", "content": f"Student message: \"{user_msg}\""},
    ]

    try:
        raw = await ollama_chat(extract_prompt)
        # Clean any markdown fences
        raw = raw.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        facts = json.loads(raw)
        if not isinstance(facts, list):
            facts = []
    except Exception as e:
        print(f"[episode extract error] {e}")
        return

    for item in facts[:3]:
        fact       = str(item.get("fact", "")).strip()
        importance = float(item.get("importance", 0.5))
        if not fact:
            continue
        try:
            embedding = await ollama_embed(fact)
        except Exception as e:
            print(f"[embed error] {e}")
            embedding = []

        episodes_col.insert_one({
            "user_id":    user_id,
            "session_id": session_id,
            "fact":       fact,
            "importance": importance,
            "embedding":  embedding,
            "created_at": now_iso(),
        })
    print(f"[episodes] stored {len(facts)} facts for {user_id}")


async def retrieve_top_episodes(user_id: str, message: str, k: int = TOP_K_EPISODES) -> list[dict]:
    """Embed the current message, do cosine similarity against stored episodes."""
    try:
        query_vec = await ollama_embed(message)
    except Exception as e:
        print(f"[episode retrieve embed error] {e}")
        return []

    docs = list(
        episodes_col.find(
            {"user_id": user_id},
            {"_id": 0, "fact": 1, "importance": 1, "embedding": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(100)
    )

    scored = []
    for doc in docs:
        vec = doc.get("embedding", [])
        if len(vec) != len(query_vec):
            continue
        sim = cosine_similarity(query_vec, vec)
        # Weight by importance
        score = sim * (0.5 + 0.5 * doc.get("importance", 0.5))
        scored.append({"fact": doc["fact"], "importance": doc["importance"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_prompt(
    short_term: list[dict],
    session_summary: Optional[str],
    lifetime_summary: Optional[str],
    episodes: list[dict],
    user_message: str,
) -> list[dict]:
    """Assemble the full message list for the Ollama chat call."""

    system_parts = [
        "You are a knowledgeable, encouraging AI study tutor.",
        "Answer clearly and helpfully. Use the memory context below to personalize your response.",
    ]

    if lifetime_summary:
        system_parts.append(f"\n## Student Lifetime Profile:\n{lifetime_summary}")

    if session_summary:
        system_parts.append(f"\n## This Session So Far:\n{session_summary}")

    if episodes:
        facts_text = "\n".join(f"- {e['fact']} (importance: {e['importance']:.2f})" for e in episodes)
        system_parts.append(f"\n## Relevant Past Facts:\n{facts_text}")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]

    # Short-term window
    for m in short_term:
        messages.append({"role": m["role"], "content": m["content"]})

    # Current user message
    messages.append({"role": "user", "content": user_message})

    return messages


# ═════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_id    = req.user_id.strip()
    session_id = (req.session_id or f"{user_id}_default").strip()
    user_msg   = req.message.strip()

    if not user_id or not user_msg:
        raise HTTPException(400, "user_id and message are required")

    # 1. Save user message
    save_message(user_id, session_id, "user", user_msg)

    # 2. Short-term memory
    short_term = get_short_term(user_id, session_id)

    # 3. Long-term summaries
    session_summary  = get_latest_summary(user_id, "session", session_id)
    lifetime_summary = get_latest_summary(user_id, "user")

    # 4. Episodic memory retrieval
    top_episodes = await retrieve_top_episodes(user_id, user_msg)

    # 5. Build prompt
    messages = build_prompt(short_term, session_summary, lifetime_summary, top_episodes, user_msg)

    # 6. Call Ollama
    try:
        reply = await ollama_chat(messages)
    except Exception as e:
        raise HTTPException(502, f"Ollama error: {e}")

    # 7. Save assistant message
    save_message(user_id, session_id, "assistant", reply)

    # 8. Extract & store episodic facts (background-style — don't block response)
    try:
        await extract_and_store_episodes(user_id, session_id, user_msg)
    except Exception as e:
        print(f"[episode store error] {e}")

    # 9. Maybe trigger summarization
    try:
        await maybe_summarize(user_id, session_id)
    except Exception as e:
        print(f"[summarize error] {e}")

    return {
        "reply": reply,
        "memory_used": {
            "short_term_count":  len(short_term),
            "short_term":        short_term,
            "session_summary":   session_summary,
            "lifetime_summary":  lifetime_summary,
            "episodic_facts":    top_episodes,
        },
    }


@app.get("/api/memory/{user_id}")
async def get_memory(user_id: str, session_id: Optional[str] = None):
    sid = session_id or f"{user_id}_default"

    last_messages = list(
        messages_col.find(
            {"user_id": user_id, "session_id": sid},
            {"_id": 0, "role": 1, "content": 1, "created_at": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(16)
    )
    last_messages = list(reversed(last_messages))

    session_summary  = get_latest_summary(user_id, "session", sid)
    lifetime_summary = get_latest_summary(user_id, "user")

    last_episodes = list(
        episodes_col.find(
            {"user_id": user_id},
            {"_id": 0, "fact": 1, "importance": 1, "created_at": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(20)
    )

    return {
        "user_id":          user_id,
        "session_id":       sid,
        "short_term":       last_messages,
        "session_summary":  session_summary,
        "lifetime_summary": lifetime_summary,
        "episodic_facts":   last_episodes,
    }


@app.get("/api/aggregate/{user_id}")
async def get_aggregate(user_id: str):
    # Daily message counts
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$addFields": {
            "date": {"$substr": ["$created_at", 0, 10]}
        }},
        {"$group": {
            "_id":   "$date",
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]
    daily_counts = list(messages_col.aggregate(pipeline))
    daily = [{"date": d["_id"], "count": d["count"]} for d in daily_counts]

    # Recent summaries
    recent_summaries = list(
        summaries_col.find(
            {"user_id": user_id},
            {"_id": 0, "scope": 1, "session_id": 1, "text": 1, "created_at": 1},
        )
        .sort("created_at", DESCENDING)
        .limit(5)
    )

    return {
        "user_id":         user_id,
        "daily_messages":  daily,
        "recent_summaries": recent_summaries,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}


# ── Static HTML (optional chat UI) ───────────────────────────────────────────
import pathlib
static_dir = pathlib.Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
