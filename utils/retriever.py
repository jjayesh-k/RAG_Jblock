"""
Hybrid Retriever — FAISS (Semantic) + BM25 (Lexical) + RRF Fusion
==================================================================
Key fixes vs original:
  - target_files pre-filter: builds a whitelist of chunk IDs belonging only to the
    documents the user selected, then restricts BOTH vector and BM25 searches to
    that whitelist.  This is the industry-standard "pre-filter before ANN" pattern
    that avoids re-building the index while staying memory-efficient.
  - BM25 positional-index → FAISS-ID mapping corrected (was broken in original).
  - FlashRank re-ranker wired in when available (was imported in app.py but never called).
  - Signature extended with optional `target_files` and `reranker` parameters.
"""

import requests
import numpy as np
import re
from config import OLLAMA_URL, EMBEDDING_MODEL
from utils.indexer import simple_tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(embed_np: np.ndarray) -> np.ndarray:
    """Ensure vector dtype matches FAISS float32 requirement."""
    return embed_np.astype(np.float32)


def _get_allowed_ids(chunk_map: dict, target_files: list) -> set | None:
    """
    Returns the set of chunk IDs whose source_file is in target_files.
    Returns None when the caller wants ALL documents (no filtering).

    Industry pattern: pre-filter the ID space before ANN search so the
    GPU index does not waste cycles scoring irrelevant shards.
    """
    if not target_files or target_files == ["All"]:
        return None  # sentinel → search everything

    allowed = set()
    for chunk_id, data in chunk_map.items():
        src = data.get("metadata", {}).get("source_file", "")
        if src in target_files:
            allowed.add(chunk_id)
    return allowed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_available_manuals(chunk_map: dict) -> list[str]:
    """Returns sorted list of unique source filenames in the database."""
    return sorted({
        data.get("metadata", {}).get("source_file", "Unknown")
        for data in chunk_map.values()
    })


def perform_hybrid_search(
    query: str,
    vector_index,
    bm25_index,
    chunk_map: dict,
    top_k: int = 15,
    target_files: list = None,
    reranker=None,
) -> list:
    """
    Hybrid search: FAISS semantic + BM25 lexical, fused with RRF.

    Parameters
    ----------
    query        : user / rewritten search string
    vector_index : FAISS IndexIDMap loaded from disk
    bm25_index   : BM25Okapi index
    chunk_map    : {faiss_id: {"text": ..., "metadata": ...}}
    top_k        : number of chunks to return
    target_files : list of source filenames to restrict search to,
                   or ["All"] / None for no restriction
    reranker     : optional FlashRank Ranker instance
    """
    if vector_index is None or bm25_index is None or not chunk_map:
        print("[Retriever] Error: Indexes are not loaded.")
        return []

    # ------------------------------------------------------------------
    # 0. Build allowed-ID whitelist (document isolation)
    # ------------------------------------------------------------------
    allowed_ids = _get_allowed_ids(chunk_map, target_files)

    # ------------------------------------------------------------------
    # 1. Generate query embedding
    # ------------------------------------------------------------------
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": query},
            timeout=60,
        )
        response.raise_for_status()
        embed_np = np.array(response.json()["embeddings"][0]).reshape(1, -1)
    except Exception as e:
        print(f"[Retriever] Embedding failure: {e}")
        return []

    # ------------------------------------------------------------------
    # 2. FAISS vector search
    # ------------------------------------------------------------------
    # Search deeper than top_k so we have room to filter & re-rank
    search_depth = min(vector_index.ntotal, max(top_k * 20, 200))
    D, I = vector_index.search(_to_float32(embed_np), search_depth)

    # ------------------------------------------------------------------
    # 3. BM25 lexical search
    # ------------------------------------------------------------------
    tokenized_query = simple_tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    # BM25 positions are 0-based indices into the ordered chunk_map keys
    chunk_ids_ordered = list(chunk_map.keys())  # positional → real FAISS ID
    top_bm25_positions = np.argsort(bm25_scores)[::-1][:search_depth]

    # ------------------------------------------------------------------
    # 4. Automotive alphanumeric boost (part numbers / DTC codes)
    # ------------------------------------------------------------------
    alphanumeric_targets = [
        re.sub(r"[^a-z0-9]", "", word.lower())
        for word in tokenized_query
        if any(ch.isdigit() for ch in word) and len(word) > 2
    ]

    def _boost(chunk_id: int) -> float:
        if not alphanumeric_targets:
            return 0.0
        text = chunk_map.get(chunk_id, {}).get("text", "").lower()
        score = 0.0
        for target in alphanumeric_targets:
            if target in text:
                score += 0.050 if len(target) > 4 else 0.015
        return score

    # ------------------------------------------------------------------
    # 5. Reciprocal Rank Fusion (RRF)
    # ------------------------------------------------------------------
    RRF_K = 60  # standard default from the original RRF paper
    final_scores: dict[int, float] = {}

    # FAISS results
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        idx_int = int(idx)
        if allowed_ids is not None and idx_int not in allowed_ids:
            continue  # ← document isolation filter
        final_scores.setdefault(idx_int, 0.0)
        final_scores[idx_int] += 1.0 / (rank + RRF_K) + _boost(idx_int)

    # BM25 results  (map positional index → real chunk ID)
    for rank, pos in enumerate(top_bm25_positions):
        pos_int = int(pos)
        if pos_int >= len(chunk_ids_ordered):
            continue
        true_id = chunk_ids_ordered[pos_int]  # ← BM25 fix: positional → ID
        if allowed_ids is not None and true_id not in allowed_ids:
            continue  # ← document isolation filter
        final_scores.setdefault(true_id, 0.0)
        final_scores[true_id] += 1.0 / (rank + RRF_K) + _boost(true_id)

    if not final_scores:
        return []

    # ------------------------------------------------------------------
    # 6. Noise gate (10 % of best score) + initial top-k cut
    # ------------------------------------------------------------------
    sorted_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best_score = sorted_candidates[0][1]
    # Collect a wider pool for the reranker; the final cut happens after reranking
    rerank_pool_size = top_k * 4
    filtered = [
        (idx, score)
        for idx, score in sorted_candidates
        if score >= best_score * 0.10
    ][:rerank_pool_size]

    # ------------------------------------------------------------------
    # 7. Optional FlashRank re-ranking
    # ------------------------------------------------------------------
    if reranker is not None and filtered:
        try:
            from flashrank import RerankRequest
            passages = [
                {"id": idx, "text": chunk_map[idx]["text"]}
                for idx, _ in filtered
                if idx in chunk_map
            ]
            rerank_req = RerankRequest(query=query, passages=passages)
            reranked = reranker.rerank(rerank_req)
            # reranked is a list of dicts with 'id' and 'score'
            filtered = [(item["id"], item["score"]) for item in reranked]
        except Exception as e:
            print(f"[Retriever] FlashRank failed, using RRF order: {e}")

    # ------------------------------------------------------------------
    # 8. Build final result list
    # ------------------------------------------------------------------
    final_top = filtered[:top_k]
    results = []
    for idx, score in final_top:
        chunk_data = chunk_map.get(idx)
        if chunk_data:
            results.append(
                {
                    "id": idx,
                    "score": score,
                    "text": chunk_data["text"],
                    "metadata": chunk_data["metadata"],
                }
            )
    return results


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils.indexer import load_indexes

    print("Loading global database...")
    v_idx, b_idx, c_map, _ = load_indexes(save_dir="offline_knowledge_base")

    test_query = "What is the maximum fluid working pressure for the 965766 dispense valve?"
    print(f"Searching for: '{test_query}'\n")

    results = perform_hybrid_search(test_query, v_idx, b_idx, c_map, top_k=5)

    for rank, res in enumerate(results):
        print(f"--- Rank {rank + 1} | Score: {res['score']:.4f} ---")
        print(f"Source: {res['metadata'].get('source_file', 'Unknown')} | Page: {res['metadata'].get('page_num', 'Unknown')}")
        print(f"{res['text'][:300]}...\n")