"""
Forge Orchestrator (main.py)
============================
1. Scans INPUT_FOLDER for PDF manuals.
2. Checks the existing database and skips already-indexed files.
3. Parses new PDFs into a master JSON ledger via parser.py.
4. Embeds and indexes the ledger into the FAISS/BM25 database via indexer.py.

Run this once per batch of new manuals.  It is safe to re-run — already
indexed files are detected and skipped automatically.
"""

import os
import glob
import time

from utils.parser import NativeMarkdownParser
from utils.indexer import build_database_from_json, load_indexes, get_indexed_files

# --- CONFIGURATION ---
INPUT_FOLDER = "input_manuals"           # Drop PDFs here
MASTER_LEDGER_FILE = "master_ledger.json"
DATABASE_FOLDER = "offline_knowledge_base"


def setup_directories() -> bool:
    """Creates INPUT_FOLDER if it doesn't exist.  Returns False if empty."""
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(
            f"[INFO] Created '{INPUT_FOLDER}'. "
            "Drop your PDFs there and run again."
        )
        return False
    return True


def run_orchestrator():
    print("=" * 50)
    print(" INITIATING FORGE PIPELINE")
    print("=" * 50)

    if not setup_directories():
        return

    # 1. Discover PDFs
    pdf_files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"[WARNING] No PDFs found in '{INPUT_FOLDER}'.")
        return
    print(f"[INFO] Found {len(pdf_files)} PDF(s) in input directory.")

    # 2. Check existing database to avoid re-parsing
    print("[PROCESS] Checking existing database for already-indexed files...")
    _, _, existing_chunk_map, _ = load_indexes(DATABASE_FOLDER)
    already_indexed = get_indexed_files(existing_chunk_map)
    if already_indexed:
        print(f"[INFO] {len(already_indexed)} file(s) already indexed — will skip.")

    # 3. Parse only new PDFs
    parser = NativeMarkdownParser()
    master_chunks = []
    new_pdfs_count = 0
    start_time = time.time()

    for idx, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"\n[{idx + 1}/{len(pdf_files)}] {filename}")

        if filename in already_indexed:
            print("  -> [SKIP] Already in database.")
            continue

        print("  -> [PARSE] Extracting pages...")
        chunks = parser.parse_pdf_to_pages(pdf_path, verbose=True)
        master_chunks.extend(chunks)
        new_pdfs_count += 1
        print(f"  -> [DONE] {len(chunks)} chunks extracted.")

    if new_pdfs_count == 0:
        print("\n[INFO] All manuals are already indexed.  Nothing to do.")
        return

    # 4. Save master ledger
    print(f"\n[INFO] Saving {len(master_chunks)} new chunks to master ledger...")
    parser.export_to_json(master_chunks, MASTER_LEDGER_FILE)

    # 5. Embed and index
    print("\n[PROCESS] Embedding and indexing...")
    build_database_from_json(MASTER_LEDGER_FILE, DATABASE_FOLDER)

    elapsed = round((time.time() - start_time) / 60, 2)
    print("\n" + "=" * 50)
    print(f" PIPELINE COMPLETE IN {elapsed} MINUTES")
    print(f" Database ready at: ./{DATABASE_FOLDER}/")
    print("=" * 50)


if __name__ == "__main__":
    run_orchestrator()