"""
Flask RAG Server — Offline Automotive Diagnostic Assistant
==========================================================
Key fixes vs original:
  - `perform_hybrid_search` now receives `reranker=RANKER` so FlashRank is
    actually used (was imported but never passed to the retriever before).
  - `target_files` is forwarded correctly to the retriever; the retriever now
    enforces document isolation at search time.
  - Query embedding normalised before FAISS search (matches index normalisation).
  - History window kept consistent: capped at 8 messages, rewriter uses last 4.
  - Minor: `/documents` endpoint deduplication fixed (was iterating chunk_map
    values correctly but doc_counts key was source_file string — preserved).
"""

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
import json
import requests
import webbrowser
from threading import Timer
import traceback

# --- CUSTOM MODULES ---
from utils.indexer import load_indexes
from utils.retriever import perform_hybrid_search, get_available_manuals
from config import LANGUAGE_MODEL, OLLAMA_URL

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
MASTER_V_INDEX = None
MASTER_B_INDEX = None
MASTER_C_MAP = None
VECTOR_DIR = "offline_knowledge_base"

# In-memory conversation history (single-user local deployment)
LOCAL_CHAT_HISTORY: list[dict] = []
# Tracks which documents were active when history was last recorded.
# Used by the query rewriter to detect document-switch mid-conversation.
ACTIVE_TARGET_FILES: list[str] = ["All"]

# FlashRank re-ranker (optional — system works without it)
RANKER = None
try:
    from flashrank import Ranker
    RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
    print("[Startup] FlashRank re-ranker loaded.")
except Exception as e:
    print(f"[Startup] FlashRank not available — using RRF order. ({e})")


def _load_master_database():
    """Loads the persistent offline automotive database into RAM on startup."""
    global MASTER_V_INDEX, MASTER_B_INDEX, MASTER_C_MAP
    print(f"[Startup] Loading database from '{VECTOR_DIR}'...")
    try:
        v_idx, b_idx, c_map, _ = load_indexes(VECTOR_DIR)
        if v_idx is not None:
            MASTER_V_INDEX = v_idx
            MASTER_B_INDEX = b_idx
            MASTER_C_MAP = c_map
            print(f"[Startup] Database ready — {v_idx.ntotal} vectors across {len(c_map)} chunks.")
        else:
            print("[Startup] WARNING: No database found. Run main.py first.")
    except Exception as e:
        print(f"[Startup] Database load failed: {e}")


def _rewrite_query(raw_query: str, target_files: list) -> str:
    """
    Uses the LLM to expand pronouns / implicit context into a self-contained
    search string.  Only fires when there is prior conversation history AND
    the active document selection has not changed since the last turn.

    Key fix: if the user switched documents (target_files differs from what
    was active during the stored history), the history is stale context from
    a different manual — injecting it would poison the search.  In that case
    we return the raw query unchanged and let FAISS find the right chunks
    purely from the user's words.
    """
    if not LOCAL_CHAT_HISTORY:
        return raw_query

    # If the active file set changed since history was recorded, skip rewriting.
    # ACTIVE_TARGET_FILES holds what was selected when the last turn was stored.
    if target_files != ACTIVE_TARGET_FILES:
        print(
            f"[QueryRewriter] Document selection changed "
            f"({ACTIVE_TARGET_FILES} → {target_files}). "
            "Skipping rewrite to avoid cross-manual context injection."
        )
        return raw_query

    # Use the last 4 messages (2 turns) — enough context, avoids prompt bloat
    recent = LOCAL_CHAT_HISTORY[-4:]
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    )

    prompt = (
        "You are a search query optimizer for a database of automotive service manuals. "
        "Given the conversation history and a new user question, rewrite the question as "
        "a precise standalone search query that includes the specific machine, part number, "
        "or DTC code being discussed.\n"
        "RULES:\n"
        "1. Inject the machine/part name from history ONLY if the new question omits it "
        "AND it is clearly still the same topic.\n"
        "2. Do NOT answer the question — only output the rewritten query.\n"
        "3. Keep the query under 20 words.\n"
        "4. If the new question seems to be about a completely different topic or manual, "
        "return it exactly as written.\n\n"
        f"History:\n{history_text}\n\n"
        f"New Question: {raw_query}\n\n"
        "Standalone Search Query:"
    )

    try:
        payload = {
            "model": LANGUAGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 60},
        }
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=45)
        resp.raise_for_status()
        rewritten = (
            resp.json()["message"]["content"]
            .strip()
            .replace('"', "")
            .replace("Here is the standalone query:", "")
            .strip()
        )
        print(f"[QueryRewriter] '{raw_query}' → '{rewritten}'")
        return rewritten or raw_query
    except Exception as e:
        print(f"[QueryRewriter] Failed ({e}) — using raw query.")
        return raw_query


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/documents", methods=["GET"])
def get_documents():
    """Returns list of loaded manuals with chunk counts for the sidebar UI."""
    if MASTER_C_MAP is None:
        return jsonify([])

    doc_counts: dict[str, int] = {}
    for chunk_data in MASTER_C_MAP.values():
        src = chunk_data.get("metadata", {}).get("source_file", "Unknown")
        doc_counts[src] = doc_counts.get(src, 0) + 1

    docs = sorted(
        [{"filename": k, "chunks": v} for k, v in doc_counts.items()],
        key=lambda x: x["filename"],
    )
    return jsonify(docs)


@app.route("/chat", methods=["POST"])
def chat():
    global MASTER_V_INDEX, MASTER_B_INDEX, MASTER_C_MAP
    global LOCAL_CHAT_HISTORY, ACTIVE_TARGET_FILES

    data = request.get_json(force=True, silent=True) or {}
    raw_query = data.get("message", "").strip()

    # Normalise target_files — must be a non-empty list
    target_files: list[str] = data.get("target_files", ["All"])
    if not isinstance(target_files, list) or len(target_files) == 0:
        target_files = ["All"]

    if not raw_query:
        return jsonify({"error": "Message is required."}), 400

    if MASTER_V_INDEX is None:
        return jsonify({"error": "Database offline. Run main.py to build the index."}), 503

    # If the user switched documents, wipe history so the rewriter starts clean.
    # Stale history from a different manual would poison the rewritten query.
    if sorted(target_files) != sorted(ACTIVE_TARGET_FILES):
        print(
            f"[Chat] Document selection changed "
            f"{ACTIVE_TARGET_FILES} → {target_files}. "
            "Clearing conversation history."
        )
        LOCAL_CHAT_HISTORY.clear()
        ACTIVE_TARGET_FILES = target_files

    # 1. Rewrite query for context continuity (skipped automatically on doc switch)
    search_query = _rewrite_query(raw_query, target_files)

    # 2. Hybrid retrieval with document isolation + optional reranking
    try:
        retrieved_chunks = perform_hybrid_search(
            query=search_query,
            vector_index=MASTER_V_INDEX,
            bm25_index=MASTER_B_INDEX,
            chunk_map=MASTER_C_MAP,
            top_k=15,
            target_files=target_files,
            reranker=RANKER,
        )
    except Exception as e:
        traceback.print_exc()
        # Stream the error so the frontend reader gets a proper 'done' and unfreezes
        def _err_gen(msg):
            yield json.dumps({"type": "error", "message": msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return Response(stream_with_context(_err_gen(f"Retrieval failed: {e}")), mimetype="application/x-ndjson")

    # No results — stream an error token instead of returning HTTP 404.
    # An HTTP 404 causes the frontend fetch to resolve but the streaming reader
    # never receives a 'done' frame, leaving the UI frozen with loading dots.
    if not retrieved_chunks:
        msg = (
            "No relevant content found in the selected manual(s) for that query. "
            "Try rephrasing or selecting additional manuals."
        )
        def _no_result_gen():
            yield json.dumps({"type": "context", "data": []}) + "\n"
            yield json.dumps({"type": "token", "content": msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return Response(stream_with_context(_no_result_gen()), mimetype="application/x-ndjson")

    # 3. Build context string from top chunks
    top_chunks = [
        (c["id"], c["text"], c["score"], c["metadata"])
        for c in retrieved_chunks
    ]
    context_str = "\n\n---\n\n".join(txt for _, txt, _, _ in top_chunks)

    # 4. Streaming LLM response generator
    def generate():
        # First frame: citation metadata
        yield json.dumps({
            "type": "context",
            "data": [
                {
                    "source": meta.get("source_file", "Unknown"),
                    "page": meta.get("page_num", "N/A"),
                }
                for _, _, _, meta in top_chunks
            ],
        }) + "\n"

        system_msg = (
            "You are an expert Automotive Diagnostic AI Assistant.\n"
            "Answer ONLY from the CONTEXT provided below. Follow these rules:\n"
            "1. Be direct and step-by-step where appropriate.\n"
            "2. Reference specific DTC codes, part numbers, torque specs, or procedures "
            "exactly as they appear in the CONTEXT.\n"
            "3. If the answer is not in the CONTEXT, reply EXACTLY with: "
            "'I don't know based on the provided manuals.' — do not guess.\n"
            "4. Do not offer generic advice or standard troubleshooting steps unless "
            "they appear verbatim in the CONTEXT.\n"
            "5. Stop writing immediately once your answer is complete."
        )

        payload = {
            "model": LANGUAGE_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {raw_query}\n\nANSWER:",
                },
            ],
            "stream": True,
            "options": {"num_predict": 512, "temperature": 0.01},
        }

        try:
            llm_resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            )
            llm_resp.raise_for_status()

            full_response = ""
            for line in llm_resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_response += content
                        yield json.dumps({"type": "token", "content": content}) + "\n"
                except json.JSONDecodeError:
                    continue

            # Persist to in-memory history and record which docs were active
            if full_response:
                LOCAL_CHAT_HISTORY.append({"role": "user", "content": raw_query})
                LOCAL_CHAT_HISTORY.append({"role": "assistant", "content": full_response})
                ACTIVE_TARGET_FILES = target_files   # keep in sync
                while len(LOCAL_CHAT_HISTORY) > 8:
                    LOCAL_CHAT_HISTORY.pop(0)

            yield json.dumps({"type": "done"}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Clears the active conversation memory and document-selection state."""
    global LOCAL_CHAT_HISTORY, ACTIVE_TARGET_FILES
    LOCAL_CHAT_HISTORY.clear()
    ACTIVE_TARGET_FILES = ["All"]
    return jsonify({"message": "Conversation history cleared."})


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_master_database()

    def _open_browser():
        webbrowser.open_new("http://127.0.0.1:8080/")

    print("Starting Offline RAG Engine on http://0.0.0.0:8080 ...")
    Timer(1.5, _open_browser).start()
    app.run(port=8080, host="0.0.0.0", debug=True, use_reloader=False)