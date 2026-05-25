"""
RAG Retrieval Diagnostic Tool
==============================
Runs a query through every stage of the pipeline and prints exactly
what is happening at each step so you can pinpoint where retrieval fails.

Usage:
    python diagnose.py

Edit the DIAGNOSTIC_CONFIG section below before running.
"""

import os
import sys
import json
import pickle
import requests
import numpy as np
import faiss

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
DATABASE_FOLDER   = "offline_knowledge_base"
TARGET_PDF        = "Robot maintenance manual.pdf"   # exact filename as stored
TEST_QUERY        = "article number CBS type GA20 nitrogen mass"
OLLAMA_URL        = "http://localhost:11434"
EMBEDDING_MODEL   = "nomic-embed-text"
TOP_K             = 15
SEARCH_DEPTH      = 400
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "─" * 70
SEP2 = "═" * 70

def hr(title=""):
    if title:
        print(f"\n{SEP}\n  {title}\n{SEP}")
    else:
        print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Load database
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 0 · Load database")

faiss_path = os.path.join(DATABASE_FOLDER, "faiss.index")
meta_path  = os.path.join(DATABASE_FOLDER, "metadata.pkl")

if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
    print(f"[FAIL] Database not found in '{DATABASE_FOLDER}'. Run main.py first.")
    sys.exit(1)

vector_index = faiss.read_index(faiss_path)

with open(meta_path, "rb") as f:
    raw = pickle.load(f)

chunk_map         = {int(k): v for k, v in raw.get("chunk_map", {}).items()}
tokenized_corpus  = raw.get("tokenized_corpus", [])

print(f"  Total vectors in FAISS index : {vector_index.ntotal}")
print(f"  Total entries in chunk_map   : {len(chunk_map)}")
print(f"  Tokenized corpus length      : {len(tokenized_corpus)}")

# Check for mismatch
if len(chunk_map) != len(tokenized_corpus):
    print(f"\n  [WARNING] chunk_map ({len(chunk_map)}) and tokenized_corpus "
          f"({len(tokenized_corpus)}) lengths differ!")
    print("  This means some chunks have no BM25 entry — BM25 results will be wrong.")
else:
    print("  [OK] chunk_map and tokenized_corpus sizes match.")

if vector_index.ntotal != len(chunk_map):
    print(f"\n  [WARNING] FAISS has {vector_index.ntotal} vectors but chunk_map "
          f"has {len(chunk_map)} entries — some IDs are orphaned.")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Inspect the target PDF's chunks
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 1 · Inspect target PDF in database")

all_sources = {}
for cid, data in chunk_map.items():
    src = data.get("metadata", {}).get("source_file", "UNKNOWN")
    all_sources.setdefault(src, []).append(cid)

print(f"  All indexed source files ({len(all_sources)} total):")
for src, ids in sorted(all_sources.items()):
    marker = " ◄ TARGET" if src == TARGET_PDF else ""
    print(f"    [{len(ids):4d} chunks]  {src}{marker}")

target_ids = all_sources.get(TARGET_PDF, [])
if not target_ids:
    print(f"\n  [FAIL] '{TARGET_PDF}' is NOT in the database at all!")
    print("  Possible causes:")
    print("    1. The filename stored in chunk_map differs from what you typed above.")
    print("    2. The PDF was never indexed — run main.py.")
    print("    3. The file was indexed under a different name (check list above).")
    sys.exit(1)

print(f"\n  [OK] '{TARGET_PDF}' found with {len(target_ids)} chunks.")
print(f"  Chunk ID range: {min(target_ids)} – {max(target_ids)}")

# Show the first 3 chunks so you can verify content was parsed correctly
print(f"\n  First 3 chunk previews (200 chars each):")
for cid in sorted(target_ids)[:3]:
    text = chunk_map[cid].get("text", "")
    page = chunk_map[cid].get("metadata", {}).get("page_num", "?")
    print(f"\n    [Chunk {cid} | Page {page}]")
    print(f"    {repr(text[:200])}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Keyword grep: does the answer text actually exist in stored chunks?
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 2 · Keyword grep across target PDF chunks")

keywords = ["GA20", "CBS", "nitrogen", "article", "hazard", "REACH", "AR", "KR 210"]
print(f"  Searching for keywords: {keywords}\n")

hits = {kw: [] for kw in keywords}
for cid in target_ids:
    text = chunk_map[cid].get("text", "").lower()
    page = chunk_map[cid].get("metadata", {}).get("page_num", "?")
    for kw in keywords:
        if kw.lower() in text:
            hits[kw].append((cid, page))

all_found = True
for kw, found in hits.items():
    if found:
        pages = sorted(set(p for _, p in found))
        print(f"  [FOUND] '{kw}' → {len(found)} chunks, pages: {pages}")
    else:
        print(f"  [MISSING] '{kw}' → NOT found in any stored chunk")
        all_found = False

if not all_found:
    print("\n  [DIAGNOSIS] Some keywords are missing from stored chunks.")
    print("  This means the PDF text was either:")
    print("    a) Not extracted (scanned/image-only PDF — needs OCR)")
    print("    b) Extracted but then stripped by the boilerplate cleaner")
    print("    c) The PDF has non-standard encoding (run STAGE 2b below)")
else:
    print("\n  [OK] All keywords found in stored text — parsing is fine.")
    print("  The problem is in RETRIEVAL, not parsing.")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2b — Show raw text for the page that should contain the answer
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 2b · Raw stored text for pages 1–20 of target PDF")

print("  (First 400 chars per page — check if meaningful text is present)\n")
for cid in sorted(target_ids):
    page = chunk_map[cid].get("metadata", {}).get("page_num", 0)
    if page > 20:
        continue
    text = chunk_map[cid].get("text", "")
    print(f"  ── Page {page:3d} | Chunk {cid} | {len(text)} chars ──")
    print(f"  {repr(text[:400])}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Embed the query and check vector similarity directly
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 3 · Query embedding + FAISS similarity scores")

print(f"  Query: '{TEST_QUERY}'")
print(f"  Embedding model: {EMBEDDING_MODEL}")

try:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": TEST_QUERY},
        timeout=60,
    )
    resp.raise_for_status()
    q_vec = np.array(resp.json()["embeddings"][0], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)
    print(f"  [OK] Query vector shape: {q_vec.shape}, norm: {np.linalg.norm(q_vec):.4f}")
except Exception as e:
    print(f"  [FAIL] Could not embed query: {e}")
    sys.exit(1)

# Search for more results than usual to see where target chunks rank
actual_depth = min(vector_index.ntotal, max(SEARCH_DEPTH, 500))
D, I = vector_index.search(q_vec, actual_depth)

# Find where target chunks appear in FAISS results
target_id_set = set(target_ids)
faiss_result_ids = [int(x) for x in I[0] if x != -1]

print(f"\n  FAISS searched {actual_depth} vectors, got {len(faiss_result_ids)} results.")
print(f"\n  Target PDF chunks in FAISS top-{actual_depth} results:")

target_hits_in_faiss = []
for rank, idx in enumerate(faiss_result_ids):
    if idx in target_id_set:
        score = float(D[0][rank])
        page  = chunk_map[idx].get("metadata", {}).get("page_num", "?")
        target_hits_in_faiss.append((rank + 1, idx, score, page))

if not target_hits_in_faiss:
    print("  [CRITICAL] No chunks from the target PDF appear in FAISS results at all!")
    print("  This means the index type is wrong or vectors were never normalised.")
    print("  Check: was the index built with IndexFlatIP or IndexFlatL2?")
else:
    print(f"  Found {len(target_hits_in_faiss)} target chunks in FAISS results.")
    print(f"  {'Rank':>6}  {'ChunkID':>8}  {'Score':>8}  {'Page':>5}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*5}")
    for rank, cid, score, page in target_hits_in_faiss[:30]:
        print(f"  {rank:>6}  {cid:>8}  {score:>8.4f}  {page:>5}")
    if len(target_hits_in_faiss) > 30:
        print(f"  ... and {len(target_hits_in_faiss) - 30} more")

    best_rank = target_hits_in_faiss[0][0]
    if best_rank > TOP_K:
        print(f"\n  [DIAGNOSIS] Best target chunk is at rank {best_rank}.")
        print(f"  With top_k={TOP_K} and search_depth={SEARCH_DEPTH}, it is being CUT OFF.")
        print(f"  Fix: increase search_depth or top_k in retriever.py.")
    else:
        print(f"\n  [OK] Best target chunk is at rank {best_rank} — within top_k.")

# Show top-10 FAISS results with their sources for comparison
print(f"\n  Top 10 FAISS results (for comparison):")
print(f"  {'Rank':>4}  {'ChunkID':>8}  {'Score':>8}  {'Page':>5}  Source")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*40}")
for rank, idx in enumerate(faiss_result_ids[:10]):
    score = float(D[0][rank])
    meta  = chunk_map.get(idx, {}).get("metadata", {})
    page  = meta.get("page_num", "?")
    src   = meta.get("source_file", "?")[:45]
    print(f"  {rank+1:>4}  {idx:>8}  {score:>8.4f}  {page:>5}  {src}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — BM25 scores for the target PDF
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 4 · BM25 lexical scores for target PDF chunks")

try:
    from rank_bm25 import BM25Okapi
    import re

    def simple_tokenize(text):
        text = re.sub(r"\[.*?\]", "", text)
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    bm25 = BM25Okapi(tokenized_corpus)
    q_tokens = simple_tokenize(TEST_QUERY)
    print(f"  Query tokens: {q_tokens}")

    scores = bm25.get_scores(q_tokens)
    chunk_ids_ordered = list(chunk_map.keys())

    # Get scores for target chunks specifically
    target_bm25 = []
    for pos, cid in enumerate(chunk_ids_ordered):
        if cid in target_id_set:
            target_bm25.append((pos, cid, float(scores[pos]),
                                chunk_map[cid].get("metadata", {}).get("page_num", "?")))

    target_bm25.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  BM25 scores for target PDF chunks (top 20):")
    print(f"  {'BM25Pos':>7}  {'ChunkID':>8}  {'Score':>8}  {'Page':>5}")
    print(f"  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*5}")
    for pos, cid, score, page in target_bm25[:20]:
        print(f"  {pos:>7}  {cid:>8}  {score:>8.4f}  {page:>5}")

    # Global BM25 top results
    top_bm25_global = np.argsort(scores)[::-1][:10]
    print(f"\n  Global BM25 top 10:")
    print(f"  {'BM25Pos':>7}  {'ChunkID':>8}  {'Score':>8}  {'Page':>5}  Source")
    print(f"  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*40}")
    for pos in top_bm25_global:
        pos_int = int(pos)
        if pos_int >= len(chunk_ids_ordered):
            continue
        cid   = chunk_ids_ordered[pos_int]
        score = float(scores[pos_int])
        meta  = chunk_map.get(cid, {}).get("metadata", {})
        page  = meta.get("page_num", "?")
        src   = meta.get("source_file", "?")[:45]
        print(f"  {pos_int:>7}  {cid:>8}  {score:>8.4f}  {page:>5}  {src}")

    zero_bm25 = sum(1 for _, _, s, _ in target_bm25 if s == 0.0)
    if zero_bm25 == len(target_bm25):
        print(f"\n  [DIAGNOSIS] ALL {len(target_bm25)} target chunks have BM25 score = 0.0")
        print("  This means the tokenized_corpus is out of sync with chunk_map.")
        print("  The BM25 index needs to be rebuilt — delete metadata.pkl and re-run main.py.")

except Exception as e:
    print(f"  [FAIL] BM25 check failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Index health check
# ──────────────────────────────────────────────────────────────────────────────
hr("STAGE 5 · Index type and normalisation check")

index_type = type(vector_index).__name__
print(f"  FAISS index type : {index_type}")

# Check if it's an IDMap wrapping an IP index (correct) or L2 (wrong for nomic)
if hasattr(vector_index, 'index'):
    inner = type(vector_index.index).__name__
    print(f"  Inner index type : {inner}")
    if "IP" in inner:
        print("  [OK] Using inner product — correct for normalised embeddings.")
    elif "L2" in inner:
        print("  [WARNING] Using L2 distance — this still works for normalised")
        print("  vectors but IP is faster and the scores are inverted (lower=better).")
        print("  If your index was built before the IndexFlatIP fix, rebuild it.")

# Sample a few vectors and check their norms
print(f"\n  Sampling 5 stored vector norms (should be ~1.0 for normalised):")
sample_ids = sorted(list(chunk_map.keys()))[:5]
sample_np  = np.array(sample_ids, dtype="int64").reshape(-1, 1)
try:
    # Reconstruct vectors from the index
    recon = np.zeros((len(sample_ids), vector_index.d), dtype="float32")
    for i, cid in enumerate(sample_ids):
        vector_index.reconstruct(cid, recon[i])
    for i, cid in enumerate(sample_ids):
        norm = float(np.linalg.norm(recon[i]))
        print(f"    Chunk {cid}: norm = {norm:.6f}")
    avg_norm = float(np.mean([np.linalg.norm(recon[i]) for i in range(len(sample_ids))]))
    if avg_norm < 0.95:
        print(f"\n  [WARNING] Average norm {avg_norm:.4f} — vectors are NOT normalised.")
        print("  Rebuild the index: delete offline_knowledge_base/ and re-run main.py.")
    else:
        print(f"\n  [OK] Average norm {avg_norm:.4f} — vectors are normalised.")
except Exception as e:
    print(f"  Could not reconstruct vectors (index may not support it): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
hr("SUMMARY")

print("""
  After running this script, look for these patterns:

  STAGE 2 shows keywords MISSING
  → The PDF text was not extracted properly.
    Cause A: Scanned/image PDF — needs OCR (add pytesseract to parser.py)
    Cause B: Boilerplate cleaner stripped too much text
    Cause C: PDF uses non-standard font encoding
    Fix: Run python -c "import fitz; d=fitz.open('your.pdf'); print(d[13].get_text())"
         to see the raw text on a specific page.

  STAGE 2 keywords FOUND but STAGE 3 shows target chunks ranked very low
  → Embedding mismatch or index built with wrong similarity metric.
    Fix: Delete offline_knowledge_base/ entirely and re-run main.py so the
         index is rebuilt with IndexFlatIP + normalised vectors.

  STAGE 3 target chunks in top results but system still returns nothing
  → The allowed_ids filter is stripping them (filename mismatch).
    Fix: Copy the exact filename shown in STAGE 1 into your TARGET_PDF above
         and compare it to what the UI sends as target_files.

  STAGE 4 all BM25 scores = 0.0
  → tokenized_corpus is out of sync (common after incremental re-indexing).
    Fix: Delete metadata.pkl and re-run main.py to rebuild BM25 corpus.

  STAGE 5 norms < 0.95
  → Old index built before normalisation fix.
    Fix: Delete offline_knowledge_base/ and re-run main.py.
""")

print(SEP2)
print("  Diagnostic complete.")
print(SEP2)