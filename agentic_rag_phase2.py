#!/usr/bin/env python3
"""
Phase 2: Agentic RAG Core Logic (Improved and Bug-Fixed, with Confidence and Caching)

This script implements the intelligent decision-making agent for the Q&A system.
It assumes Phase 1 (data ingestion, chunking, embedding, and index building)
has already been executed, and the necessary index files are available in the 'cache' directory.

This version adds:
- Confidence Score calculation for generated answers.
- Redis-based caching layer for query-answer pairs using vector similarity.
"""

import os
import re
import json
import logging
import time
import pickle
import requests
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# New Imports for Redis Caching
import hashlib
import redis        
from redis.commands.search.query import Query
from redis.commands.search.field import TagField, VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType


# ---------------- Globals & Constants ----------------

CACHE_DIR = "cache"
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_index.bin")
FAISS_METADATA_FILE = os.path.join(CACHE_DIR, "faiss_metadata.json")
BM25_INDEX_FILE = os.path.join(CACHE_DIR, "bm25_index.pkl")

# Gemini LLM Configuration for Agent Reasoning and Generation
AGENT_LLM_MODEL = "gemini-2.0-flash"
AGENT_LLM_TEMP_LOW = 0.0     # Very low for classification
AGENT_LLM_TEMP_MEDIUM = 0.2  # Low for summarization/answer generation
AGENT_LLM_MAX_OUTPUT_TOKENS_SHORT = 10   # For classification signals
AGENT_LLM_MAX_OUTPUT_TOKENS_LONG = 1024  # For generated answers/summaries

# Agent Decision Keywords (no typos, must match prompts exactly)
DECISION_IRRELEVANT = "IRRELEVANT"
DECISION_GENERAL_QA = "GENERAL_QA"
DECISION_COMPANY_SPECIFIC = "COMPANY_SPECIFIC"
INSUFFICIENT_CONTEXT_SIGNAL = "INSUFFICIENT_CONTEXT"
NO_ANSWER_IN_CONTEXT_SIGNAL = "NO_ANSWER_IN_CONTEXT"
NO_RELEVANT_SEARCH_RESULTS_SIGNAL = "NO_RELEVANT_SEARCH_RESULTS"

# Serper AI Configuration
SERPER_API_KEY = None  # Loaded from .env
SERPER_SEARCH_URL = "https://google.serper.dev/search"

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None) # If your Redis requires a password
REDIS_INDEX_NAME = "rag_cache_index"

# Caching Thresholds
CACHING_SIMILARITY_THRESHOLD = 0.97 # Cosine similarity for query caching

# Confidence Score Parameters
CONFIDENCE_SEMANTIC_WEIGHT = 0.6  # Weight for query-answer semantic similarity
CONFIDENCE_RETRIEVAL_WEIGHT = 0.4  # Weight for average retrieval score (for RAG path)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- NLTK Resource Check ---
def ensure_nltk_resources():
    """Ensures NLTK punkt tokenizer is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")


# --- Environment Variable Loading ---
def load_env_vars():
    """Loads API keys from .env file and configures Gemini."""
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY")

    global SERPER_API_KEY
    SERPER_API_KEY = serper_key

    if not gemini_key:
        logger.error("Missing GEMINI_API_KEY in .env file. Exiting.")
        exit(1)
    if not serper_key:
        logger.warning("Missing SERPER_API_KEY in .env file. Internet search will be disabled.")

    genai.configure(api_key=gemini_key)
    logger.info("Environment variables loaded and Gemini configured.")


# ---------------- Helper Utilities ----------------

def get_gemini_response(
    prompt: str,
    model_name: str = AGENT_LLM_MODEL,
    temperature: float = AGENT_LLM_TEMP_LOW,
    max_output_tokens: int = AGENT_LLM_MAX_OUTPUT_TOKENS_SHORT
) -> str:
    """
    Centralized function to get a response from Gemini.
    Returns an empty string on error or no content.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        # Navigate the response structure safely
        if response and getattr(response, "candidates", None):
            candidate = response.candidates[0]
            if getattr(candidate.content, "parts", None):
                part = candidate.content.parts[0]
                text = getattr(part, "text", "").strip()
                return text
        logger.warning(f"Gemini returned no valid content. Full response: {response}")
        return ""
    except Exception as e:
        logger.error(f"Error calling Gemini API for model '{model_name}': {e}")
        return ""


def get_gemini_embeddings(
    texts: list[str],
    task_type: str
) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using Gemini's text-embedding-004 model.
    Includes retry logic and zero-vector fallback for failures.
    """
    if not texts:
        return []

    embeddings = []
    embedding_dim = 768  # text-embedding-004 dimension

    for batch_start in range(0, len(texts), 50):  # BATCH_SIZE = 50
        batch_end = min(batch_start + 50, len(texts))
        batch_texts = texts[batch_start:batch_end]
        retries = 0
        success = False

        while retries < 3 and not success:  # MAX_RETRIES = 3
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch_texts,
                    task_type=task_type
                )
                embeddings.extend(response["embedding"])
                success = True
            except Exception as e:
                logger.warning(
                    f"Embedding batch {batch_start//50} failed (attempt {retries+1}): {e}"
                )
                retries += 1
                time.sleep(2 * retries)  # RETRY_DELAY = 2 seconds * attempt

        if not success:
            logger.error(f"Embedding batch {batch_start//50} permanently failed. Using zero vectors.")
            for _ in range(len(batch_texts)):
                embeddings.append([0.0] * embedding_dim)

    return embeddings

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0 # Handle zero vectors to avoid division by zero
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ---------------- Phase 1 Artifact Loaders ----------------

def load_faiss_index_and_metadata(
    index_path: str = FAISS_INDEX_FILE,
    metadata_path: str = FAISS_METADATA_FILE
):
    """
    Loads a FAISS index and its associated metadata from disk.
    Returns (index, metadata_dict) or (None, None) on failure.
    """
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        logger.warning(f"FAISS index '{index_path}' or metadata '{metadata_path}' not found.")
        return None, None

    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"FAISS index and metadata loaded from '{index_path}' and '{metadata_path}'")
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading FAISS index or metadata: {e}")
        return None, None


def load_bm25_index(bm25_path: str = BM25_INDEX_FILE):
    """
    Loads the BM25 index from disk.
    Returns the BM25Okapi object or None on failure.
    """
    if not os.path.exists(bm25_path):
        logger.warning(f"BM25 index file '{bm25_path}' not found.")
        return None

    try:
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)
        logger.info(f"BM25 index loaded from '{bm25_path}'")
        return bm25_index
    except Exception as e:
        logger.error(f"Error loading BM25 index: {e}")
        return None


# ---------------- Hybrid Retrieval ----------------

def hybrid_retrieval(
    query: str,
    faiss_index,
    faiss_metadata: dict,
    bm25_index,
    top_k: int = 5,
    alpha: float = 0.7
) -> list[dict]:
    """
    Performs hybrid retrieval combining FAISS (dense) and BM25 (sparse) search.
    Requires loaded faiss_index, faiss_metadata, bm25_index.
    Returns a list of top_k metadata dicts with 'score' and (optionally) 'parent_content'.
    """
    if faiss_index is None or faiss_metadata is None:
        logger.error("FAISS index or metadata not loaded. Cannot perform dense retrieval.")
        return []

    # 1) Dense retrieval via FAISS
    try:
        query_emb_list = get_gemini_embeddings([query], task_type="RETRIEVAL_QUERY")
        q_emb = np.array(query_emb_list[0], dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb /= norm
            distances, indices = faiss_index.search(q_emb, top_k * 3)
            dense_scores = distances[0]    # shape (top_k*3,)
            dense_indices = indices[0]     # shape (top_k*3,)
        else:
            logger.warning("Query embedding is zero vector. Skipping dense retrieval.")
            dense_scores = np.array([], dtype=np.float32)
            dense_indices = np.array([], dtype=int)
    except Exception as e:
        logger.error(f"Failed dense retrieval: {e}")
        dense_scores = np.array([], dtype=np.float32)
        dense_indices = np.array([], dtype=int)

    # 2) Sparse retrieval via BM25
    if bm25_index is not None:
        tokenized_query = [
            w for w in word_tokenize(query.lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2
        ]
        if tokenized_query:
            sparse_scores = np.array(bm25_index.get_scores(tokenized_query))
            top_sparse_indices = np.argsort(sparse_scores)[::-1][: top_k * 3]
        else:
            logger.warning("Tokenized query is empty. Skipping sparse retrieval.")
            sparse_scores = np.array([])
            top_sparse_indices = np.array([])
    else:
        sparse_scores = np.array([])
        top_sparse_indices = np.array([])

    # 3) Combine scores
    combined_scores = {}
    max_dense = float(np.max(dense_scores)) if dense_scores.size > 0 else 1.0
    max_sparse = float(np.max(sparse_scores)) if sparse_scores.size > 0 else 1.0

    # Accumulate dense contributions
    for idx, score in zip(dense_indices, dense_scores):
        key = str(idx)
        if key not in faiss_metadata:
            continue
        norm_score = float(score) / max_dense if max_dense > 0 else 0.0
        combined_scores[key] = combined_scores.get(key, 0.0) + alpha * norm_score

    # Accumulate sparse contributions
    for idx in top_sparse_indices:
        key = str(idx)
        if key not in faiss_metadata:
            continue
        norm_score = float(sparse_scores[idx]) / max_sparse if max_sparse > 0 else 0.0
        combined_scores[key] = combined_scores.get(key, 0.0) + (1 - alpha) * norm_score

    if not combined_scores:
        logger.warning("No valid results found in hybrid retrieval.")
        return []

    # 4) Sort and pick top_k
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[: top_k]
    # Build a parent_content map for O(1) lookups
    parent_content_map = {
        entry["chunk_id"]: entry.get("content", "")
        for entry in faiss_metadata.values()
        if entry.get("parent")
    }

    results = []
    for idx_str, combined_score in sorted_items:
        entry = faiss_metadata[idx_str].copy()
        entry["score"] = float(combined_score)
        parent_id = entry.get("parent")
        if parent_id:
            entry["parent_content"] = parent_content_map.get(parent_id, "")
        results.append(entry)

    logger.info(f"Hybrid retrieval returned {len(results)} chunks for query '{query[:50]}...'")
    return results


# ---------------- Confidence Scoring ----------------

def calculate_confidence_score(
    query_embedding: np.ndarray,
    answer_text: str,
    retrieved_chunks: list[dict],
    path_taken: str,
    raw_query: str
) -> float:
    """
    Calculates a confidence score for the generated answer (0.0 to 1.0).
    The method varies based on the path taken.
    """
    confidence = 0.0
    
    # Ensure embeddings are 1D numpy arrays
    query_embedding = np.asarray(query_embedding).flatten()

    # 1. Semantic Similarity between Query and Answer
    # This assesses how well the answer semantically aligns with the query.
    semantic_sim = 0.0
    if answer_text and query_embedding.shape[0] > 0:
        answer_embedding = get_gemini_embeddings([answer_text], task_type="SEMANTIC_SIMILARITY")[0]
        answer_embedding = np.asarray(answer_embedding).flatten()
        if answer_embedding.shape[0] > 0:
            semantic_sim = cosine_similarity(query_embedding, answer_embedding)
            # Normalize to 0-1 range: (sim + 1) / 2
            semantic_sim = (semantic_sim + 1) / 2 
    
    logger.info(f"Confidence: Semantic similarity (query-answer): {semantic_sim:.2f}")

    if path_taken == "Company_Specific_RAG_Success":
        # Combine semantic similarity with retrieval quality
        retrieval_scores = [c["score"] for c in retrieved_chunks if "score" in c]
        if retrieval_scores:
            avg_retrieval_score = np.mean(retrieval_scores)
            # Retrieval scores are already normalized 0-1 by hybrid_retrieval
            logger.info(f"Confidence: Average retrieval score: {avg_retrieval_score:.2f}")
            confidence = (CONFIDENCE_SEMANTIC_WEIGHT * semantic_sim) + \
                         (CONFIDENCE_RETRIEVAL_WEIGHT * avg_retrieval_score)
        else:
            confidence = semantic_sim * 0.7 # If no scores, rely mostly on semantic sim but slightly penalize
        
    elif path_taken == "Internet_Search_Success":
        # For internet search, confidence is primarily semantic similarity
        confidence = semantic_sim
        
    elif path_taken == "General_QA_LLM_Knowledge":
        # For general Q&A, if LLM responded, we can assume a decent confidence based on its direct knowledge
        confidence = semantic_sim * 0.8 + 0.2 # Give a baseline if it generated something coherent

    else: # Irrelevant, No_Answer_Found
        confidence = 0.0

    # Ensure confidence is within [0, 1] bounds
    confidence = max(0.0, min(1.0, confidence))
    logger.info(f"Calculated confidence for '{path_taken}': {confidence:.2f}")
    return confidence


# ---------------- Redis Caching Layer ----------------

import redis
import numpy as np
import json
import logging
import time # For generating unique keys
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TagField,
    TextField,
    VectorField,
    NumericField
)
from redis.commands.search.index_definition import IndexDefinition, IndexType

logger = logging.getLogger(__name__)

class RedisCacheManager:
    def __init__(self, host='localhost', port=6379, db=0, password=None, index_name='rag_cache_index', vector_dimension=768):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.index_name = index_name
        self.vector_dimension = vector_dimension
        self.client = None
        self._connect_with_retry()  # Changed to retry connection
        self._create_index()  # Create or recreate the index

    def _connect_with_retry(self, max_retries=3, retry_delay=2):
        """Establishes connection with retry logic"""
        for attempt in range(max_retries):
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=False  # Keep binary data for vectors
                )
                self.client.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}/{self.db}")
                return
            except redis.exceptions.ConnectionError as e:
                logger.warning(f"Connection attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(retry_delay)
        logger.error("Could not connect to Redis after multiple attempts")
        self.client = None

    def _create_index(self):
        """Creates or recreates the RediSearch index for the cache."""
        if not self.client:
            logger.warning("Redis client not connected. Cannot create index.")
            return

        # Try to drop existing index if exists
        try:
            self.client.ft(self.index_name).dropindex()
            logger.info(f"Dropped existing index: {self.index_name}")
        except Exception:
            logger.info(f"No existing index to drop: {self.index_name}")

        # Define schema with CORRECTED field names
        schema = (
            TextField("$.query_text", as_name="query_text"),
            TextField("$.answer_json", as_name="answer_json"),
            TagField("$.path_taken", as_name="path_taken"),  # Changed to TagField
            NumericField("$.confidence_score", as_name="confidence_score"),  # Changed to NumericField
            VectorField(
                "$.embedding",
                "HNSW",  # Use approximate search for better performance
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dimension,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": 1000,
                },
                as_name="embedding_vector",
            ),
        )

        # Define index with JSON type
        definition = IndexDefinition(prefix=["rag_cache:"], index_type=IndexType.JSON)
        
        try:
            self.client.ft(self.index_name).create_index(fields=schema, definition=definition)
            logger.info(f"Created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")

    def store_in_cache(self, query: str, embedding: np.ndarray, answer_data: dict):
        """Stores an item in Redis cache with PROPER embedding handling"""
        if not self.client:
            return

        # Ensure embedding is 1D array
        embedding = np.asarray(embedding).flatten().astype(np.float32)
        
        cache_entry = {
            "query_text": query,
            "answer_json": json.dumps(answer_data),
            "path_taken": answer_data.get("path_taken", ""),
            "confidence_score": float(answer_data.get("confidence_score", 0.0)),
            "embedding": embedding.tolist()  # Store as list of floats
        }
        
        key = f"rag_cache:{hashlib.sha256(query.encode()).hexdigest()}"
        
        try:
            self.client.json().set(key, "$", cache_entry)
            logger.info(f"Cached response for: {query[:50]}...")
        except Exception as e:
            logger.error(f"Caching error: {e}")

    def retrieve_from_cache(self, query_embedding: np.ndarray, threshold: float) -> dict | None:
        """Retrieves from cache with FIXED vector search"""
        if not self.client:
            return None

        try:
            # Prepare query embedding
            query_embedding = np.asarray(query_embedding).flatten().astype(np.float32)
            query_bytes = query_embedding.tobytes()
            
            # Corrected query parameters
            base_query = f"@embedding_vector:[VECTOR_RANGE $range $vec]=>{{$YIELD_DISTANCE_AS: dist}}"
            query = (
                Query(base_query)
                .return_fields("dist", "answer_json", "path_taken", "confidence_score")
                .sort_by("dist")
                .dialect(2)
                .paging(0, 1)
            )
            
            params = {
                "vec": query_bytes,
                "range": 1 - threshold  # Convert similarity to distance
            }
            
            results = self.client.ft(self.index_name).search(query, params)
            
            if results.total > 0:
                doc = results.docs[0]
                distance = float(doc.dist)
                similarity = 1 - distance
                
                if similarity >= threshold:
                    logger.info(f"Cache hit: {similarity:.4f} >= {threshold}")
                    answer_data = json.loads(doc.answer_json)
                    answer_data["cached"] = True
                    answer_data["cache_similarity"] = similarity
                    return answer_data
                    
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None


# ---------------- Phase 2 Agentic Core Logic ----------------

# Prompt Templates

INITIAL_QUERY_PROCESSOR_PROMPT = (
    "You are an intelligent assistant designed to categorize user queries.\n"
    "Your task is to determine two things about a user's query:\n"
    "1.  **Relevance to Company Policies:** Is the query related to company employee policies, procedures, benefits, HR, or internal company operations?\n"
    "2.  **Knowledge Type:** Can the query be answered with general knowledge that a large language model (LLM) would typically know, or does it require specific internal company information?\n\n"
    "**IMPORTANT:** Your response MUST be exactly one of the following keywords, with no additional text:\n"
    f"-   {DECISION_IRRELEVANT}\n"
    f"-   {DECISION_GENERAL_QA}\n"
    f"-   {DECISION_COMPANY_SPECIFIC}\n\n"
    "Examples:\n"
    "Query: \"What is the capital of France?\"\n"
    f"Response: {DECISION_IRRELEVANT}\n\n"
    "Query: \"What are good ways to improve productivity at work?\"\n"
    f"Response: {DECISION_GENERAL_QA}\n\n"
    "Query: \"How many vacation days do I get?\"\n"
    f"Response: {DECISION_COMPANY_SPECIFIC}\n\n"
    "Query: \"Can I use sick leave for mental health days?\"\n"
    f"Response: {DECISION_COMPANY_SPECIFIC}\n\n"
    "Query: \"How do I reset my password?\"\n"
    f"Response: {DECISION_COMPANY_SPECIFIC}\n\n"
    "Query: \"What is a good diet for weight loss?\"\n"
    f"Response: {DECISION_IRRELEVANT}\n\n"
    "**Now categorize this query:**\n"
    "Query: {query}\n"
    "Response: "
)

INTERNAL_CONTEXT_ANSWER_PROMPT = (
    "You are an expert Q&A assistant. You will be given a user’s question plus a set of internal document chunks.\n"
    "Each chunk has metadata (chunk_id, source, page_number) and the chunk’s full content.\n\n"
    "Your task is to:\n"
    "1. Carefully read *all* the provided chunks.\n"
    "2. If any chunk (or combination of chunks) contains sufficient information to answer the user’s question *completely and accurately*, compose a concise answer and **include citations** in the form\n"
    "   `(Source: <filename>, Page <n>)` or `(ChunkID: <chunk_id>)` right after any fact you report.\n"
    "3. If none of the provided chunks contain information that can answer this question, respond exactly with:\n"
    f"   `{NO_ANSWER_IN_CONTEXT_SIGNAL}`\n\n"
    "Do NOT hallucinate or use information outside the provided chunks.\n\n"
    "Chunks (in JSON-like format). Each entry is:\n"
    "    {\n"
    "      \"chunk_id\": \"<chunk_id>\",\n"
    "      \"source\": \"<filename>\",\n"
    "      \"page_number\": <page_no>,\n"
    "      \"content\": \"<full paragraph or chunk text>\"\n"
    "    }\n\n"
    "Here are the chunks:\n"
    "---\n"
    "{chunks_json}\n"
    "---\n\n"
    "Question: {query}\n\n"
    "Answer (with citations or exactly \"{NO_ANSWER_IN_CONTEXT_SIGNAL}\"):"
)

INTERNET_SEARCH_SUMMARIZER_PROMPT = (
    "You are an information extraction and summarization agent.\n"
    "Given a user's question and a list of raw internet search results, your task is to:\n"
    "1.  Identify the most relevant snippets that address the user's question.\n"
    "2.  Synthesize those snippets into a concise, coherent context directly related to the question.\n"
    "3.  If no relevant information is found in the search results, respond EXACTLY with:\n"
    f"   \"{NO_RELEVANT_SEARCH_RESULTS_SIGNAL}\"\n\n"
    "Do NOT add any other text.\n\n"
    "Question: {query}\n\n"
    "Search Results:\n"
    "---\n"
    "{search_results_text}\n"
    "---\n\n"
    "Extracted and Summarized Context (or \"{NO_RELEVANT_SEARCH_RESULTS_SIGNAL}\"):"
)


def classify_query(query: str) -> str:
    """
    Agent 1: Determines if query is IRRELEVANT, GENERAL_QA, or COMPANY_SPECIFIC.
    Returns one of: DECISION_IRRELEVANT, DECISION_GENERAL_QA, DECISION_COMPANY_SPECIFIC.
    """
    logger.info(f"Agent 1: Classifying query: '{query}'")
    prompt = INITIAL_QUERY_PROCESSOR_PROMPT.replace("{query}", query)
    decision = get_gemini_response(
        prompt,
        temperature=AGENT_LLM_TEMP_LOW,
        max_output_tokens=AGENT_LLM_MAX_OUTPUT_TOKENS_SHORT
    ).strip().upper()

    if decision == DECISION_IRRELEVANT:
        logger.info("Agent 1 decision: IRRELEVANT")
        return DECISION_IRRELEVANT
    if decision == DECISION_GENERAL_QA:
        logger.info("Agent 1 decision: GENERAL_QA")
        return DECISION_GENERAL_QA
    if decision == DECISION_COMPANY_SPECIFIC:
        logger.info("Agent 1 decision: COMPANY_SPECIFIC")
        return DECISION_COMPANY_SPECIFIC

    # If LLM returned something unexpected, log and default to COMPANY_SPECIFIC
    logger.warning(f"Agent 1 unexpected response: '{decision}'. Defaulting to COMPANY_SPECIFIC.")
    return DECISION_COMPANY_SPECIFIC


def evaluate_and_answer(query: str, retrieved_chunks: list[dict]):
    """
    Attempts to answer the query from the provided internal context chunks.
    Returns a tuple (answer_text, used_sources).

    - If the LLM can answer entirely from context, answer_text is the answer
      (with citation markers), and used_sources is a list of the specific
      chunk_ids that were cited by the LLM.

    - If the LLM returns '{NO_ANSWER_IN_CONTEXT_SIGNAL}', answer_text is exactly
      that signal, and used_sources is [].
    """
    logger.info(f"Agent 2: Attempting to answer from internal context for query: '{query}'")
    if not retrieved_chunks:
        logger.info("No retrieved chunks available → returning NO_ANSWER_IN_CONTEXT.")
        return NO_ANSWER_IN_CONTEXT_SIGNAL, []

    # Build list of chunks for LLM
    chunks_for_llm = []
    for chunk in retrieved_chunks:
        chunks_for_llm.append({
            "chunk_id":    chunk["chunk_id"],
            "source":      chunk["source"],
            "page_number": chunk["page_number"],
            "content":     chunk["content"]
        })

    # Serialize as JSON
    chunks_json_text = json.dumps(chunks_for_llm, ensure_ascii=False)

    prompt = INTERNAL_CONTEXT_ANSWER_PROMPT.replace("{chunks_json}", chunks_json_text) \
                                          .replace("{query}", query) \
                                          .replace("{NO_ANSWER_IN_CONTEXT_SIGNAL}", NO_ANSWER_IN_CONTEXT_SIGNAL)

    response = get_gemini_response(
        prompt,
        temperature=AGENT_LLM_TEMP_MEDIUM,
        max_output_tokens=AGENT_LLM_MAX_OUTPUT_TOKENS_LONG
    ).strip()

    if response.upper() == NO_ANSWER_IN_CONTEXT_SIGNAL:
        logger.info("Agent 2: LLM signaled NO_ANSWER_IN_CONTEXT.")
        return NO_ANSWER_IN_CONTEXT_SIGNAL, []

    # Extract cited chunk_ids or source/page citations
    used_sources = []

    # (ChunkID: abc) patterns
    for match in re.findall(r"\(ChunkID:\s*([^)]+)\)", response):
        used_sources.append(match.strip())

    # (Source: filename, Page n) patterns
    for match in re.findall(r"\(Source:\s*([^,]+),\s*Page\s*(\d+)\)", response):
        filename = match[0].strip()
        page_no = match[1].strip()
        used_sources.append(f"{filename}|Page{page_no}")

    used_sources = list(dict.fromkeys(used_sources))  # deduplicate

    logger.info(f"Agent 2: LLM answered from context, citing {used_sources}")
    return response, used_sources


def perform_internet_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Performs an internet search using Serper AI.
    Returns a list of {title, link, snippet} dicts.
    """
    if not SERPER_API_KEY:
        logger.error("SERPER_API_KEY not found. Internet search disabled.")
        return []

    logger.info(f"Performing internet search for query: '{query}' via Serper AI.")
    try:
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {
            "q": query,
            "num": min(num_results, 100),
            "gl": "us",
            "hl": "en"
        }
        response = requests.post(SERPER_SEARCH_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        results = response.json()

        search_items = []
        for item in results.get("organic", []):
            search_items.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })

        # Prepend answerBox or knowledgeGraph if present
        if "answerBox" in results and results["answerBox"].get("snippet"):
            ab = results["answerBox"]
            search_items.insert(0, {
                "title": ab.get("title", "Answer Box"),
                "link": ab.get("link", ""),
                "snippet": ab.get("snippet", "")
            })
        elif "knowledgeGraph" in results and results["knowledgeGraph"].get("snippet"):
            kg = results["knowledgeGraph"]
            search_items.insert(0, {
                "title": kg.get("title", "Knowledge Graph"),
                "link": kg.get("link", ""),
                "snippet": kg.get("snippet", "")
            })

        logger.info(f"Internet search returned {len(search_items)} items.")
        return search_items

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error during Serper AI search: {e}")
        return []
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"JSON decode error from Serper AI response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during Serper AI search: {e}")
        return []


def summarize_search_results(query: str, search_results: list[dict]) -> str:
    """
    Summarizes internet search results via Gemini, returning a context string or NO_RELEVANT_SEARCH_RESULTS_SIGNAL.
    """
    if not search_results:
        logger.info("No search results to summarize. Returning NO_RELEVANT_SEARCH_RESULTS_SIGNAL.")
        return NO_RELEVANT_SEARCH_RESULTS_SIGNAL

    # Concatenate up to 5 results into a text
    combined_text = ""
    for item in search_results[:5]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        combined_text += f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n\n"

    # Truncate by sentences to approx 1500 words
    sentences = sent_tokenize(combined_text)
    truncated, word_count = [], 0
    for sent in sentences:
        wc = len(sent.split())
        if word_count + wc > 1500:
            break
        truncated.append(sent)
        word_count += wc
    truncated_text = " ".join(truncated)
    if word_count < len(combined_text.split()):
        logger.warning("Search results truncated for summarization due to length.")

    prompt = INTERNET_SEARCH_SUMMARIZER_PROMPT.replace("{query}", query) \
                                            .replace("{search_results_text}", truncated_text) \
                                            .replace("{NO_RELEVANT_SEARCH_RESULTS_SIGNAL}", NO_RELEVANT_SEARCH_RESULTS_SIGNAL)

    summary = get_gemini_response(
        prompt,
        temperature=AGENT_LLM_TEMP_MEDIUM,
        max_output_tokens=AGENT_LLM_MAX_OUTPUT_TOKENS_LONG
    ).strip()

    if summary.upper() == NO_RELEVANT_SEARCH_RESULTS_SIGNAL:
        logger.info("Summarizer returned NO_RELEVANT_SEARCH_RESULTS_SIGNAL.")
        return NO_RELEVANT_SEARCH_RESULTS_SIGNAL

    logger.info("Summarizer returned a valid context.")
    return summary


def answer_query_agentic(query: str, faiss_index, faiss_metadata: dict, bm25_index) -> dict:
    """
    Orchestrates the agentic RAG process with a single pass of internal context to LLM.
    Returns { 'answer': str, 'path_taken': str, 'sources': list[dict] }.
    """
    path_taken = "Unknown"
    final_answer = "I'm sorry, I couldn't find a relevant answer."
    sources = []

    logger.info(f"\n--- Processing Query: '{query}' ---")

    # Step 1: Classify query
    agent1_decision = classify_query(query)

    if agent1_decision == DECISION_IRRELEVANT:
        path_taken = "Irrelevant"
        final_answer = "I'm sorry, your question is outside the scope of company policy knowledge."
        return {"answer": final_answer, "path_taken": path_taken, "sources": sources}

    if agent1_decision == DECISION_GENERAL_QA:
        path_taken = "General_QA_LLM_Knowledge"
        prompt = f"Answer the following general question concisely:\n\n{query}"
        answer = get_gemini_response(
            prompt,
            temperature=AGENT_LLM_TEMP_MEDIUM,
            max_output_tokens=AGENT_LLM_MAX_OUTPUT_TOKENS_LONG
        ).strip()
        final_answer = answer or "I apologize, but I couldn't generate an answer to that general question."
        return {"answer": final_answer, "path_taken": path_taken, "sources": sources}

    # At this point: COMPANY_SPECIFIC
    path_taken = "Company_Specific_RAG_Attempt"
    logger.info("Agent Path: Company_Specific_RAG_Attempt")

    # Step 2: Hybrid Retrieval
    retrieved_chunks = hybrid_retrieval(
        query=query,
        faiss_index=faiss_index,
        faiss_metadata=faiss_metadata,
        bm25_index=bm25_index,
        top_k=5
    )

    # Step 3: Single-pass internal context → LLM
    internal_answer, cited_chunk_ids = evaluate_and_answer(query, retrieved_chunks)

    if internal_answer != NO_ANSWER_IN_CONTEXT_SIGNAL:
        path_taken = "Company_Specific_RAG_Success"
        final_answer = internal_answer

        # Build sources list for CLI
        for chunk in retrieved_chunks:
            # Check if this chunk's content was cited by the LLM
            # (Note: this is a basic check. For perfect accuracy, LLM should output exact chunk_id citations)
            is_cited = False
            for cited_id in cited_chunk_ids:
                if cited_id == chunk["chunk_id"] or (
                    "|" in cited_id and cited_id.startswith(f"{chunk['source']}|Page{chunk['page_number']}")
                ):
                    is_cited = True
                    break
            
            # Only add to sources if actually used by LLM or highly relevant
            if is_cited or chunk.get("score", 0) > 0.5: # Example: also add if retrieval score is high
                snippet = chunk["content"]
                snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
                sources.append({
                    "type": "Internal Document",
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "content": snippet
                })

        return {"answer": final_answer, "path_taken": path_taken, "sources": sources}

    # Internal context insufficient → fallback to Internet Search
    path_taken = "Company_Specific_Internet_Search_Attempt"
    logger.info("Agent Path: Company_Specific_Internet_Search_Attempt")

    search_results = perform_internet_search(query, num_results=5)
    internet_context = summarize_search_results(query, search_results)

    if internet_context != NO_RELEVANT_SEARCH_RESULTS_SIGNAL:
        path_taken = "Internet_Search_Success"
        logger.info("Agent Path: Internet_Search_Success")

        prompt = (
            "You are an expert Q&A assistant. The user’s question is:\n\n"
            f"{query}\n\n"
            "Below are summarized internet snippets relevant to the question. "
            "Answer the question accurately, and add citations in this format:\n"
            "(Internet Source: <title or snippet source>).\n\n"
            "Summarized Internet Context:\n"
            f"{internet_context}\n\n"
            "Answer with citations:"
        )

        answer = get_gemini_response(
            prompt,
            temperature=AGENT_LLM_TEMP_MEDIUM,
            max_output_tokens=AGENT_LLM_MAX_OUTPUT_TOKENS_LONG
        ).strip()
        final_answer = answer or "I apologize, but I couldn't generate an answer from internet search results."

        # Add internet sources for CLI
        for item in search_results:
            snippet_text = item.get("snippet", "")
            snippet_display = snippet_text[:200] + "..." if len(snippet_text) > 200 else snippet_text
            sources.append({
                "type": "Internet Search",
                "title":  item.get("title", ""),
                "link":   item.get("link", ""),
                "snippet": snippet_display
            })

        return {"answer": final_answer, "path_taken": path_taken, "sources": sources}

    # No answer anywhere
    path_taken = "No_Answer_Found"
    final_answer = "I’m sorry, I couldn’t find a relevant answer in our internal documents or via internet search."
    return {"answer": final_answer, "path_taken": path_taken, "sources": sources}


# ---------------- Wrapper for Caching and Confidence ----------------

def answer_query_agentic_with_cache(
    query: str,
    faiss_index,
    faiss_metadata: dict,
    bm25_index,
    cache_manager: RedisCacheManager
) -> dict:
    # Get embeddings with CORRECT dimension
    query_embedding = np.array(
        get_gemini_embeddings([query], task_type="RETRIEVAL_QUERY")[0],
        dtype=np.float32
    ).flatten()
    logger.info("Searching in the cache...")
    # Try cache retrieval
    cached = cache_manager.retrieve_from_cache(query_embedding, CACHING_SIMILARITY_THRESHOLD)
    if cached:
        logger.info("Cache hit!")
        return cached
    logger.info("Cache Missed:( Proceeding with normal flow...")
    # Process query normally
    response = answer_query_agentic(query, faiss_index, faiss_metadata, bm25_index)

    logger.info("Checking for possible cache storing...")
    # Add confidence score
    internal_chunks = [
        c for c in response.get("sources", []) 
        if c.get("type") == "Internal Document"
    ]
    response["confidence_score"] = calculate_confidence_score(
        query_embedding,
        response.get("answer", ""),
        internal_chunks,
        response.get("path_taken", ""),
        query
    )

    # Cache if meets criteria
    cacheable_paths = [
        "Company_Specific_RAG_Success",
        "Internet_Search_Success",
        "General_QA_LLM_Knowledge"
    ]
    if (response["path_taken"] in cacheable_paths and 
        response["confidence_score"] > 0.5):
        logger.info("Cache is being stored...Done :)")
        cache_manager.store_in_cache(query, query_embedding, response)
    else:
        logger.info("Cache not suitable to store..")
    
    return response



# ---------------- CLI Entry Point ----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Agentic RAG Core Logic")
    parser.add_argument("--query", action="store_true", help="Run in interactive query mode")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of results to return in internal retrieval")
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        exit(0)

    # Load environment and NLTK resources
    load_env_vars()
    ensure_nltk_resources()

    # --- Initialize Redis Cache Manager ---
    redis_cache_manager = RedisCacheManager(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        index_name=REDIS_INDEX_NAME,
        vector_dimension=768  # Gemini embedding dimension
    )

    # Check if Redis connection was successful. If not, caching will be disabled internally.
    if not redis_cache_manager.client:
        logger.warning("Caching layer is disabled due to Redis connection/index creation issues.")

    # Load Phase 1 artifacts
    logger.info("Loading Phase 1 artifacts (FAISS index, metadata, BM25 index)...")
    faiss_index, faiss_metadata = load_faiss_index_and_metadata()
    bm25_index = load_bm25_index()

    if faiss_index is None or faiss_metadata is None:
        logger.critical("FAISS index or metadata not found. Please run Phase 1 '--build' first.")
        exit(1)
    if bm25_index is None:
        logger.warning("BM25 index not found. Hybrid retrieval will operate in dense-only mode.")

    logger.info("All Phase 1 artifacts loaded successfully. Agent is ready.")

    print("\n=== Agentic RAG Q&A System CLI ===")
    print("Type your query and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        # Call the new wrapper function
        response = answer_query_agentic_with_cache(
            query,
            faiss_index,
            faiss_metadata,
            bm25_index,
            redis_cache_manager # Pass the manager instance
        )

        print(f"\nAnswer: {response['answer']}")
        print(f"Path Taken: {response['path_taken']}")
        print(f"Confidence Score: {response.get('confidence_score', 'N/A'):.2f}") # Display confidence

        if response["sources"]:
            print("Sources:")
            for i, src in enumerate(response["sources"], start=1):
                if src["type"] == "Internal Document":
                    print(f"  {i}. [Internal] {src['source']} (Page {src['page_number']}): {src['content']}")
                else:
                    print(f"  {i}. [Internet] {src['title']} - Link: {src['link']} - Snippet: {src['snippet']}")
        print("-" * 80)
