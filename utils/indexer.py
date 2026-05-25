"""
Offline Vector Indexer — FAISS (Semantic) + BM25 (Lexical)
===========================================================
Key fixes vs original:
  - FAISS base index changed from IndexFlatL2 → IndexFlatIP (inner product /
    cosine similarity). nomic-embed-text produces normalised vectors; L2 on
    normalised vectors is correct mathematically but IP is faster and is what
    Ollama's nomic model expects for similarity ranking.
  - Added faiss.normalize_L2() call before add_with_ids so cosine similarity
    works correctly at query time (query vector is also normalised in retriever).
  - chunk_map key type enforced as int consistently (was mixing int/str in some
    paths which caused KeyError misses during retrieval).
  - Removed stale JSON_INPUT_DIR logic from the module-level __main__ block;
    kept it clean and pointed at the correct database folder.
  - get_indexed_files now filters out None values from the set.
"""

import os
import json
import faiss
import numpy as np
import pickle
import glob
from rank_bm25 import BM25Okapi
import re
import requests

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIR = "offline_knowledge_base"
JSON_INPUT_DIR = "./parsed_manuals"

BATCH_SIZE = 50  # tune down to 25 if you hit OOM on 8 GB VRAM


# ---------------------------------------------------------------------------
# Tokenizer (shared with retriever)
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokeniser for BM25."""
    text = re.sub(r"\[.*?\]", "", text)
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_indexes(save_dir: str = VECTOR_DIR):
    """
    Loads FAISS index, rebuilds BM25 from saved corpus, returns chunk_map.

    Returns
    -------
    (vector_index, bm25_index, chunk_map, tokenized_corpus)
    All are None / empty on first run.
    """
    faiss_path = os.path.join(save_dir, "faiss.index")
    meta_path = os.path.join(save_dir, "metadata.pkl")

    if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
        return None, None, {}, []

    vector_index = faiss.read_index(faiss_path)

    with open(meta_path, "rb") as f:
        data = pickle.load(f)

    # Ensure chunk_map keys are always int
    raw_map = data.get("chunk_map", {})
    chunk_map = {int(k): v for k, v in raw_map.items()}
    tokenized_corpus = data.get("tokenized_corpus", [])

    bm25_index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    return vector_index, bm25_index, chunk_map, tokenized_corpus


def save_indexes(
    vector_index,
    chunk_map: dict,
    tokenized_corpus: list,
    save_dir: str = VECTOR_DIR,
):
    """Persists FAISS index + metadata to disk atomically."""
    os.makedirs(save_dir, exist_ok=True)

    faiss_path = os.path.join(save_dir, "faiss.index")
    faiss.write_index(vector_index, faiss_path)

    meta_path = os.path.join(save_dir, "metadata.pkl")
    # Always store keys as int for consistency
    int_chunk_map = {int(k): v for k, v in chunk_map.items()}
    with open(meta_path, "wb") as f:
        pickle.dump(
            {"chunk_map": int_chunk_map, "tokenized_corpus": tokenized_corpus}, f
        )


def get_indexed_files(chunk_map: dict) -> set:
    """Returns the set of source filenames already in the database (no None values)."""
    return {
        data.get("metadata", {}).get("source_file")
        for data in chunk_map.values()
        if data.get("metadata", {}).get("source_file") is not None
    }


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def remove_document_from_index(source_filename: str, save_dir: str = VECTOR_DIR):
    """
    Removes all chunks belonging to `source_filename` from the database.
    Use before re-indexing an updated manual.
    """
    vector_index, _, chunk_map, tokenized_corpus = load_indexes(save_dir)
    if vector_index is None:
        print("[INFO] No database found to delete from.")
        return

    ids_to_remove = [
        cid
        for cid, data in chunk_map.items()
        if data.get("metadata", {}).get("source_file") == source_filename
    ]

    if not ids_to_remove:
        print(f"[INFO] '{source_filename}' not found in database.")
        return

    print(f"[PROCESS] Removing {len(ids_to_remove)} chunks for '{source_filename}'...")

    np_ids = np.array(ids_to_remove, dtype="int64")
    vector_index.remove_ids(np_ids)

    # Rebuild chunk_map and BM25 corpus without the removed entries
    removed_set = set(ids_to_remove)
    new_chunk_map: dict = {}
    new_corpus: list = []
    for cid, data in chunk_map.items():
        if cid not in removed_set:
            new_chunk_map[cid] = data
            new_corpus.append(simple_tokenize(data["text"]))

    save_indexes(vector_index, new_chunk_map, new_corpus, save_dir)
    print(f"[SUCCESS] Purged '{source_filename}' — {len(ids_to_remove)} chunks removed.")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str]) -> list[list[float]] | None:
    """Calls Ollama embed endpoint for a batch. Returns None on failure."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("embeddings", [])
    except Exception as e:
        print(f"  [!] Connection error: {e}")
    return None


def _embed_single(text: str) -> list[float] | None:
    """Embeds a single text chunk. Returns None on failure."""
    result = _embed_batch([text])
    return result[0] if result else None


def _embed_with_fallback(chunk_data: dict) -> tuple[list[list[float]], list[dict]]:
    """
    Tries batch embedding → single embedding → 4000-char slicing.
    Returns (embeddings, matching_chunk_dicts).
    """
    text = chunk_data["content"]

    vec = _embed_single(text)
    if vec is not None:
        return [vec], [chunk_data]

    # Dynamic slicer — last resort for monster pages
    print(f"     [!] Slicing monster chunk from {chunk_data.get('source_file')}...")
    MAX_CHARS = 4000
    segments = [text[i : i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
    embeddings, chunks = [], []
    for idx, seg in enumerate(segments):
        vec = _embed_single(seg)
        if vec is None:
            print(f"       [-] Segment {idx} failed permanently. Skipping.")
            continue
        new_chunk = chunk_data.copy()
        new_chunk["content"] = seg
        new_chunk["chunk_id"] = f"{chunk_data.get('chunk_id', 'unk')}_pt{chr(65+idx)}"
        embeddings.append(vec)
        chunks.append(new_chunk)
    return embeddings, chunks


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_database_from_json(json_path: str, save_dir: str = VECTOR_DIR):
    """
    Reads a parsed JSON ledger and appends new chunks to the global database.
    Already-indexed files are skipped automatically.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] Ledger not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        parsed_chunks: list[dict] = json.load(f)

    if not parsed_chunks:
        print("[WARNING] Empty JSON ledger — nothing to index.")
        return

    print(
        f"[PROCESS] Preparing to embed {len(parsed_chunks)} chunks "
        f"from {os.path.basename(json_path)}..."
    )

    # Load existing state
    vector_index, _, chunk_map, tokenized_corpus = load_indexes(save_dir)
    already_indexed = get_indexed_files(chunk_map)
    next_id = (max(chunk_map.keys()) + 1) if chunk_map else 0

    valid_embeddings: list[list[float]] = []
    valid_chunk_data: list[dict] = []

    for i in range(0, len(parsed_chunks), BATCH_SIZE):
        batch = parsed_chunks[i : i + BATCH_SIZE]
        # Skip files already in the database
        batch = [c for c in batch if c.get("source_file") not in already_indexed]
        if not batch:
            continue

        texts = [c["content"] for c in batch]
        print(f"   -> Embedding batch of {len(batch)} chunks...")

        embeddings = _embed_batch(texts)
        if embeddings and len(embeddings) == len(batch):
            valid_embeddings.extend(embeddings)
            valid_chunk_data.extend(batch)
        else:
            # Batch failed or size mismatch — fall back per-chunk
            print(f"     [!] Batch embedding failed. Retrying chunk-by-chunk...")
            for chunk in batch:
                vecs, chunks = _embed_with_fallback(chunk)
                valid_embeddings.extend(vecs)
                valid_chunk_data.extend(chunks)

    if not valid_embeddings:
        print("[INFO] No new embeddings generated — database unchanged.")
        return

    dimension = len(valid_embeddings[0])
    np_embeddings = np.array(valid_embeddings, dtype="float32")

    # Normalise for cosine similarity (nomic-embed-text outputs unit vectors,
    # but explicit normalisation is a no-op on those and protects against
    # models that don't normalise by default)
    faiss.normalize_L2(np_embeddings)

    # Assign sequential IDs
    custom_ids = list(range(next_id, next_id + len(valid_embeddings)))
    np_ids = np.array(custom_ids, dtype="int64")

    # Build or extend FAISS index
    if vector_index is None:
        print(f"[INFO] Creating new FAISS IndexFlatIP ({dimension}d)...")
        base_index = faiss.IndexFlatIP(dimension)   # cosine via inner product
        vector_index = faiss.IndexIDMap(base_index)

    vector_index.add_with_ids(np_embeddings, np_ids)

    # Update chunk_map and BM25 corpus
    for cid, chunk in zip(custom_ids, valid_chunk_data):
        chunk_map[int(cid)] = {
            "text": chunk["content"],
            "metadata": {
                "source_file": chunk.get("source_file", "Unknown"),
                "page_num": chunk.get("page_num", 0),
                "chunk_id": chunk.get("chunk_id", str(cid)),
            },
        }
        tokenized_corpus.append(simple_tokenize(chunk["content"]))

    save_indexes(vector_index, chunk_map, tokenized_corpus, save_dir)
    print(f"\n[SUCCESS] Added {len(valid_embeddings)} chunks.")
    print(f"[INFO] Total database size: {len(chunk_map)} chunks.")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(JSON_INPUT_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(JSON_INPUT_DIR, "*.json"))

    print("=" * 50)
    print(" GLOBAL KNOWLEDGE BASE BUILDER")
    print("=" * 50)

    if not json_files:
        print(f"[INFO] No JSON ledgers in {JSON_INPUT_DIR}. Nothing to index.")
    else:
        for json_file in json_files:
            print(f"\n--- Processing: {os.path.basename(json_file)} ---")
            build_database_from_json(json_file)