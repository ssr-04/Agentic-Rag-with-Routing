# Agentic RAG System - Phase 2: Core Logic

This repository contains the core logic for Phase 2 of an Agentic Retrieval-Augmented Generation (RAG) system. Building upon the data processing and indexing completed in Phase 1, this script implements an intelligent agent capable of classifying user queries, retrieving relevant information from internal company documents (via hybrid search), leveraging general LLM knowledge, performing internet searches when necessary, generating concise answers, calculating a confidence score for the answer, and utilizing a Redis cache for frequently asked or similar questions.

## Features

*   **Agentic Query Classification:** Determines if a query is Irrelevant, General Q&A, or Company-Specific.
*   **Hybrid Retrieval:** Combines semantic search (FAISS) and keyword search (BM25) for robust retrieval from internal document indexes.
*   **LLM Integration (Gemini):** Uses a large language model for query classification, generating answers from retrieved context, summarizing internet search results, and answering general knowledge questions.
*   **Internal RAG:** Answers Company-Specific questions using information found in the loaded internal document indexes.
*   **Internet Search Fallback:** If internal documents are insufficient for a Company-Specific query (and Serper AI is configured), it performs a targeted internet search.
*   **Internet Context Summarization:** Uses the LLM to synthesize relevant information from raw internet search results.
*   **Confidence Scoring:** Calculates a score (0.0 to 1.0) indicating the system's confidence in the generated answer, based on factors like query-answer semantic similarity and retrieval quality.
*   **Redis Caching:** Implements a vector similarity-based caching layer using Redis (with RediSearch) to store and retrieve answers for similar previous queries, reducing latency and API costs.
*   **CLI Interface:** Provides a simple command-line interface for interactive querying.

## Prerequisites

Before running this script, ensure you have the following:

1.  **Python 3.7+:** Installed on your system.
2.  **Required Libraries:** Install the necessary Python packages (see Installation).
3.  **`.env` File:** A file named `.env` in the project root containing your API keys.
    *   `GEMINI_API_KEY`: Required for LLM interactions. Get one from the [Google AI Studio](https://aistudio.google.com/).
    *   `SERPER_API_KEY`: **Optional.** Required only if you want to enable internet search fallback. Get one from [Serper AI](https://serper.dev/).
    *   `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`: Optional. Redis connection details, defaults to `localhost:6379/0`.
    *   `REDIS_PASSWORD`: Optional. If your Redis instance requires authentication.
4.  **NLTK Punkt Tokenizer:** The script will attempt to download this automatically if not found.
5.  **Redis Server with RediSearch:** A running Redis server with the RediSearch module installed and enabled. The script will attempt to connect to it and create the necessary index.
6.  **Phase 1 Artifacts:** You **must** have successfully run Phase 1 to generate the necessary index files in the `./cache` directory:
    *   `cache/faiss_index.bin` (FAISS index file)
    *   `cache/faiss_metadata.json` (FAISS metadata mapping index IDs to chunk info)
    *   `cache/bm25_index.pkl` (Serialized BM25 index)

## Installation

1.  Clone this repository (if you haven't already).
2.  Navigate to the project directory in your terminal.
3.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    *(Create a `requirements.txt` file with the following contents if it doesn't exist):*
    ```
    google-generativeai
    faiss-cpu  # or faiss-gpu for GPU support
    rank-bm25
    nltk
    python-dotenv
    requests
    redis
    numpy
    scikit-learn # for ENGLISH_STOP_WORDS (though can be replaced)
    ```
4.  Create a `.env` file in the root directory and add your API keys and optional Redis details:

    ```env
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    SERPER_API_KEY=YOUR_SERPER_API_KEY # Optional
    REDIS_HOST=localhost # Optional, default
    REDIS_PORT=6379 # Optional, default
    REDIS_DB=0 # Optional, default
    # REDIS_PASSWORD=your_redis_password # Uncomment if needed
    ```
5.  Ensure your Redis server is running and accessible from your environment, with the RediSearch module loaded.
6.  **Crucially, run Phase 1** to build the necessary indexes if you haven't already:
    ```bash
    python phase1_build.py --build # Replace with the actual Phase 1 script name and command
    ```

## Configuration

The script's behavior can be configured by modifying the constants defined at the beginning of the file:

*   **`CACHE_DIR`**, **`FAISS_INDEX_FILE`**, **`FAISS_METADATA_FILE`**, **`BM25_INDEX_FILE`**: Paths to the Phase 1 artifacts.
*   **`AGENT_LLM_MODEL`**, **`AGENT_LLM_TEMP_LOW`**, **`AGENT_LLM_TEMP_MEDIUM`**, **`AGENT_LLM_MAX_OUTPUT_TOKENS_SHORT`**, **`AGENT_LLM_MAX_OUTPUT_TOKENS_LONG`**: Gemini LLM settings for different tasks.
*   **`DECISION_IRRELEVANT`**, etc., **`INSUFFICIENT_CONTEXT_SIGNAL`**, etc.: Keywords used by the agent for decision making and communication signals. **Do not change these unless you modify the LLM prompts accordingly.**
*   **`SERPER_SEARCH_URL`**: Serper AI endpoint URL.
*   **`REDIS_HOST`**, **`REDIS_PORT`**, **`REDIS_DB`**, **`REDIS_PASSWORD`**, **`REDIS_INDEX_NAME`**: Redis connection and index details. The index name is used by RediSearch.
*   **`CACHING_SIMILARITY_THRESHOLD`**: The cosine similarity threshold (0.0 to 1.0) for a query embedding to be considered a cache hit for a previously stored query. Higher values mean stricter matching.
*   **`CONFIDENCE_SEMANTIC_WEIGHT`**, **`CONFIDENCE_RETRIEVAL_WEIGHT`**: Weights (summing to 1.0) for the two components of the confidence score calculation when using the RAG path.

## How to Run

Execute the script with the `--query` argument to start the interactive CLI:

```bash
python agentic_rag_phase2.py --query
