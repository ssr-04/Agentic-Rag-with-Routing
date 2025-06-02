#!/usr/bin/env python3
"""
Enhanced RAG Pipeline with Paragraph‐Level Chunking
"""

import os
import re
import json
import logging
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import time
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import faiss
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pickle

# ---------------- Globals & Constants ----------------
PDF_FOLDER = "docs"
CACHE_DIR = "cache"
CHUNKS_CACHE_FILE = os.path.join(CACHE_DIR, "chunks.json")
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings.json")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_index.bin")
FAISS_METADATA_FILE = os.path.join(CACHE_DIR, "faiss_metadata.json")
BM25_INDEX_FILE = os.path.join(CACHE_DIR, "bm25_index.pkl")

MAX_TOKENS_PER_CHUNK = 1000    # Now allows up to ~1000 words per chunk
MIN_CHUNK_LENGTH     = 50
GEMINI_MODEL         = "models/text-embedding-004"
EMBEDDING_DIM        = 768
BATCH_SIZE           = 50
MAX_RETRIES          = 3
RETRY_DELAY          = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------- Helper Utilities ----------------
def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")


def create_cache_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_env_vars():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("Missing GEMINI_API_KEY in .env file")
        exit(1)
    genai.configure(api_key=gemini_key)


# ---------------- PDF Extraction & Cleaning ----------------
def clean_text(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE) # To remove number only lines
    text = re.sub(r"(?i)page\s+\d+", "", text) #page numbers
    text = re.sub(r"\n+", "\n", text).strip() #consecutive new lines 
    return text


def extract_paragraphs(pdf_path: str) -> list[dict]:
    """
    Open each PDF page with PyMuPDF, extract the entire page text, 
    split on double‐newline boundaries to form paragraphs, and return 
    a list of dicts:
        {
          "source": <filename>,
          "page_number": <int>,
          "text": <one whole paragraph>,
          "block_type": "paragraph"
        }
    """
    paragraphs = []
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        for para in re.split(r"\n\s*\n", cleaned):
            para = para.strip()
            if len(para.split()) < MIN_CHUNK_LENGTH:
                continue
            paragraphs.append({
                "source":      filename,
                "page_number": page_num,
                "text":        para,
                "block_type":  "paragraph"
            })

    logger.info(f"Extracted {len(paragraphs)} paragraphs from {filename}")
    return paragraphs



# ---------------- Advanced Chunking (Paragraph‐Level) ----------------
def hierarchical_chunking(blocks: list[dict]) -> list[dict]:
    """
    Paragraph‐level chunking:
      - If block["text"] ≤ MAX_TOKENS_PER_CHUNK words, emit as one chunk
      - Otherwise, split at sentence boundaries into ≤ MAX_TOKENS_PER_CHUNK words
    """
    chunks = []
    chunk_counter = 0

    for block in blocks:
        text   = block["text"]
        source = block["source"]
        page   = block["page_number"]

        words = text.split()

        # 1) If paragraph is short, emit entire paragraph:
        if len(words) <= MAX_TOKENS_PER_CHUNK:
            chunks.append({
                "chunk_id":     f"{source}_c{chunk_counter}",
                "source":       source,
                "page_number":  page,
                "block_type":   "paragraph",
                "content":      text,
                "parent":       None
            })
            chunk_counter += 1
            continue

        # 2) Otherwise, split large paragraph into sentences, grouping ≤ MAX_TOKENS_PER_CHUNK words
        sentences = sent_tokenize(text)
        buffer_sentences = []

        for sent in sentences:
            joined = " ".join(buffer_sentences + [sent])
            if len(joined.split()) > MAX_TOKENS_PER_CHUNK:
                # Flush out current buffer
                chunk_text = " ".join(buffer_sentences).strip()
                if chunk_text:
                    chunks.append({
                        "chunk_id":     f"{source}_c{chunk_counter}",
                        "source":       source,
                        "page_number":  page,
                        "block_type":   "paragraph",
                        "content":      chunk_text,
                        "parent":       None
                    })
                    chunk_counter += 1
                buffer_sentences = [sent]
            else:
                buffer_sentences.append(sent)

        # Flush leftover sentences at end of paragraph
        if buffer_sentences:
            chunk_text = " ".join(buffer_sentences).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id":     f"{source}_c{chunk_counter}",
                    "source":       source,
                    "page_number":  page,
                    "block_type":   "paragraph",
                    "content":      chunk_text,
                    "parent":       None
                })
                chunk_counter += 1

    logger.info(f"Created {len(chunks)} paragraph-level chunks")
    return chunks



# ---------------- Embedding with Gemini (Batched + Retry) ----------------
def get_gemini_embeddings_batch(texts: list[str], task_type: str) -> list[list[float]]:
    embeddings = []
    failed_indices = []

    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(texts))
        batch_texts = texts[batch_start:batch_end]
        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            try:
                response = genai.embed_content(
                    model=GEMINI_MODEL,
                    content=batch_texts,
                    task_type=task_type
                )
                embeddings.extend(response["embedding"])
                success = True
            except Exception as e:
                logger.warning(
                    f"Batch {batch_start//BATCH_SIZE} failed (attempt {retries+1}): {e}"
                )
                retries += 1
                time.sleep(RETRY_DELAY * retries)

        if not success:
            # Record failed indices so we can fill zero vectors later
            failed_indices.extend(range(batch_start, batch_end))

    for idx in failed_indices:
        logger.error(f"Using zero vector for failed chunk index {idx}")
        embeddings.insert(idx, [0.0] * EMBEDDING_DIM)

    return embeddings


# ---------------- Hybrid Indexing (FAISS + BM25) ----------------
def build_hybrid_index(chunks: list[dict]):
    contents = [chunk["content"] for chunk in chunks]
    logger.info(f"Generating embeddings for {len(contents)} chunks...")
    raw_embs = get_gemini_embeddings_batch(contents, task_type="RETRIEVAL_DOCUMENT")

    embeddings_np = np.array(raw_embs, dtype=np.float32)
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_np /= norms

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_np)
    faiss.write_index(index, FAISS_INDEX_FILE)
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Build FAISS metadata
    faiss_metadata = {}
    for idx, chunk in enumerate(chunks):
        faiss_metadata[str(idx)] = {
            "chunk_id":     chunk["chunk_id"],
            "source":       chunk["source"],
            "page_number":  chunk["page_number"],
            "content":      chunk["content"],
            "block_type":   chunk["block_type"],
            "parent":       chunk["parent"]
        }

    with open(FAISS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(faiss_metadata, f, indent=2)

    # Build BM25 index
    logger.info("Building BM25 index...")
    tokenized_contents = []
    for content in contents:
        tokens = [
            w for w in word_tokenize(content.lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2
        ]
        tokenized_contents.append(tokens)

    if len(tokenized_contents) == 0:
        logger.error("No valid documents to build BM25 index")
        return

    bm25_index = BM25Okapi(tokenized_contents)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_index, f)
    logger.info("BM25 index saved")

    logger.info("Hybrid index (FAISS + BM25) built successfully")


# ---------------- Hybrid Retrieval ----------------
def hybrid_retrieval(query: str, top_k: int = 5, alpha: float = 0.7) -> list[dict]:
    if not(os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE)):
        logger.error("FAISS index or metadata file missing")
        return []

    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_METADATA_FILE, "r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    bm25_index = None
    if os.path.exists(BM25_INDEX_FILE):
        try:
            with open(BM25_INDEX_FILE, "rb") as f:
                bm25_index = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            bm25_index = None
    else:
        logger.warning("BM25 index file not found. Falling back to dense-only retrieval.")

    # Dense retrieval (FAISS)
    try:
        response = genai.embed_content(
            model=GEMINI_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        q_emb = np.array(response["embedding"], dtype=np.float32).reshape(1, -1)
        q_emb /= np.linalg.norm(q_emb)

        distances, indices = faiss_index.search(q_emb, top_k * 3)
        dense_scores  = distances[0]
        dense_indices = indices[0]
    except Exception as e:
        logger.error(f"Failed dense retrieval: {e}")
        dense_scores = np.array([], dtype=np.float32)
        dense_indices = np.array([], dtype=int)

    # Sparse retrieval (BM25)
    if bm25_index is not None:
        tokenized_query = [
            w for w in word_tokenize(query.lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2
        ]
        sparse_scores = np.array(bm25_index.get_scores(tokenized_query))
        top_sparse_indices = np.argsort(sparse_scores)[::-1][: top_k * 3]
    else:
        sparse_scores = np.array([])
        top_sparse_indices = np.array([])

    # Combine scores
    combined_scores = {}
    max_dense = float(np.max(dense_scores)) if dense_scores.size > 0 else 1.0
    max_sparse = float(np.max(sparse_scores)) if sparse_scores.size > 0 else 1.0

    for idx, score in zip(dense_indices, dense_scores):
        if idx < 0 or str(idx) not in faiss_metadata:
            continue
        norm_score = float(score) / max_dense if max_dense > 0 else 0.0
        combined_scores[str(idx)] = combined_scores.get(str(idx), 0.0) + alpha * norm_score

    for idx in top_sparse_indices:
        if idx < 0 or idx >= len(faiss_metadata):
            continue
        norm_score = float(sparse_scores[idx]) / max_sparse if max_sparse > 0 else 0.0
        combined_scores[str(idx)] = combined_scores.get(str(idx), 0.0) + (1 - alpha) * norm_score

    if not combined_scores:
        logger.warning("No valid results found in hybrid retrieval")
        return []

    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[: top_k]
    results = []
    for idx_str, combined_score in sorted_items:
        entry = faiss_metadata[idx_str].copy()
        entry["score"] = float(combined_score)
        if entry.get("parent"):
            parent_id = entry["parent"]
            parent_entry = next(
                (v for v in faiss_metadata.values() if v["chunk_id"] == parent_id),
                None
            )
            if parent_entry:
                entry["parent_content"] = parent_entry["content"]
        results.append(entry)

    return results


# ---------------- Pipeline Orchestration ----------------
def build_pipeline():
    create_cache_dirs()
    ensure_nltk_resources()
    load_env_vars()

    all_blocks = []
    for filename in os.listdir(PDF_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_FOLDER, filename)
        try:
            paras = extract_paragraphs(pdf_path)
            all_blocks.extend(paras)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    if not all_blocks:
        logger.error("No valid PDF blocks extracted. Exiting.")
        return

    chunks = hierarchical_chunking(all_blocks)
    with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to '{CHUNKS_CACHE_FILE}'")

    build_hybrid_index(chunks)
    logger.info("Pipeline build complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RAG Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="Build the entire pipeline")
    group.add_argument("--query", action="store_true", help="Run in query mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.build:
        build_pipeline()

    elif args.query:
        load_env_vars()
        if not (os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE)):
            logger.error("Index files missing. Please run with --build first.")
            exit(1)

        print("\n=== Enhanced RAG Retrieval CLI ===")
        print("Type your query and press Enter. Type 'exit' or 'quit' to stop.\n")

        while True:
            query = input("Query> ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            results = hybrid_retrieval(query, top_k=args.top_k)
            if not results:
                print("No results found.\n")
                continue

            for i, res in enumerate(results, start=1):
                print(f"\n[{i}] Source : {res['source']} (Page {res['page_number']})")
                print(f"Score  : {res['score']:.4f}")
                print(f"Content: {res['content']}")
                if res.get("parent_content"):
                    print(f"Parent Context: {res['parent_content']}")
                print("-" * 80)
