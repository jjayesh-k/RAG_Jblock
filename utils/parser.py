"""
Air-Gapped pymupdf4llm Parser  (1 Page = 1 Chunk)
==================================================
Key fixes vs original:
  - _fix_broken_tables: guard against empty split_cells before calling max(),
    which caused ZeroDivisionError / ValueError on pages with bare pipe chars.
  - Broader encoding artifact cleanup (adds \u0000–\u0008 range strip and
    common PDF garbage characters beyond just \x02/\x03).
  - parse_pdf_to_pages: doc is now explicitly closed after use (fitz.Document
    holds a file handle — important for batch processing many PDFs).
  - export_to_json: explicit UTF-8 BOM-free output (was already utf-8, kept).
"""

import pymupdf4llm
import fitz  # PyMuPDF
import re
import os
import json
from typing import List, Dict


class NativeMarkdownParser:
    def __init__(self):
        print("Booting Native Markdown Parser (1 page = 1 chunk)...")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_encoding_artifacts(self, text: str) -> str:
        """Strips common PDF encoding garbage before any other processing."""
        # Remove C0 control characters (except tab, newline, carriage return)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Remove known pymupdf4llm encoding artefacts
        text = text.replace("\ufffd", "").replace("\u0002", "").replace("\u0003", "")
        # Remove soft-hyphen and zero-width chars
        text = text.replace("\u00ad", "").replace("\u200b", "").replace("\u200c", "")
        return text

    def _clean_boilerplate(self, text: str) -> str:
        """Removes repetitive manual header/footer artifacts."""
        text = re.sub(r"Page:\s*\d+\s*of\s*\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Prepared by:.*?(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Approved by:.*?(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Copyright.*?(?:Ltd\.|Inc\.|LLC\.?)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _fix_broken_tables(self, text: str) -> str:
        """
        Explodes squashed <br>-separated table cells into proper Markdown rows.
        Guard against empty rows to prevent max() ValueError.
        """
        lines = text.split("\n")
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and "<br>" in stripped:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if not cells:
                    fixed_lines.append(line)
                    continue

                split_cells = [c.split("<br>") for c in cells]
                max_items = max((len(c) for c in split_cells), default=0)
                if max_items == 0:
                    fixed_lines.append(line)
                    continue

                # Pad shorter cells
                for c in split_cells:
                    while len(c) < max_items:
                        c.append("")

                for i in range(max_items):
                    row = "|" + "".join(f" {c[i].strip()} |" for c in split_cells)
                    fixed_lines.append(row)
            else:
                fixed_lines.append(line)

        return "\n".join(fixed_lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_pdf_to_pages(self, pdf_path: str, verbose: bool = True) -> List[Dict]:
        """
        Parses a PDF file into one chunk per page.

        Returns a list of dicts:
            {chunk_id, source_file, page_num, content}
        """
        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            return []

        filename = os.path.basename(pdf_path)
        if verbose:
            print(f"Processing: {filename}")

        doc = fitz.open(pdf_path)
        try:
            md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        except Exception as e:
            print(f"pymupdf4llm failed on {filename}: {e}")
            doc.close()
            return []

        page_chunks: List[Dict] = []

        for i, md_data in enumerate(md_pages):
            page_num = i + 1
            raw_page = doc[i]

            # 1. Clean encoding artifacts first, then formatting
            raw_md = self._clean_encoding_artifacts(md_data.get("text", ""))
            fixed = self._fix_broken_tables(raw_md)
            clean = self._clean_boilerplate(fixed)

            if not clean.strip():
                if verbose:
                    print(f"   -> Skipping page {page_num} (blank after cleaning)")
                continue

            # 2. Tombstone pages with diagrams/schematics
            has_images = bool(raw_page.get_images(full=True))
            has_drawings = len(raw_page.get_drawings()) > 15
            if has_images or has_drawings:
                clean += (
                    f"\n\n> **[SYSTEM NOTE: Visual diagram or schematic detected on "
                    f"page {page_num}. Refer to the original manual for visual confirmation.]**"
                )

            page_chunks.append(
                {
                    "chunk_id": f"{filename}_p{page_num}",
                    "source_file": filename,
                    "page_num": page_num,
                    "content": clean.strip(),
                }
            )

        doc.close()  # release file handle

        if verbose:
            print(f"Extracted {len(page_chunks)} clean chunks from {filename}.")
        return page_chunks

    def export_to_json(self, chunks: List[Dict], output_file: str = "parsed_manuals.json"):
        """Saves extracted chunks to a UTF-8 JSON ledger."""
        print(f"Saving {len(chunks)} chunks to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
        print("JSON export complete.")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    secure_pdf_path = r"Endure Dispense Valve 965766 308876EN-L.pdf"
    output_ledger = r"parsed_output.json"

    parser = NativeMarkdownParser()
    extracted = parser.parse_pdf_to_pages(secure_pdf_path)
    if extracted:
        parser.export_to_json(extracted, output_ledger)