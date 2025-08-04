# Folder Structure Analysis Report

## Summary
This project implements an Agentic Retrieval-Augmented Generation (RAG) system that processes user queries, retrieves relevant information from internal company documents, and generates answers with confidence scoring and caching. The codebase is organized in a simple, phase-based structure with a clear separation between the data processing/indexing phase (Phase-1) and the query processing/answering phase (Phase-2).

## Directory Layout
- `/`: Root directory containing the main Phase-2 implementation and README
  - `agentic_rag_phase2.py`: Core implementation of the agentic RAG system with confidence scoring and Redis caching
  - `README.md`: Project documentation explaining features, prerequisites, installation, and configuration
- `/Phase-1/`: Directory containing Phase-1 implementation
  - `phase1_build.py`: Implementation for document processing, chunking, embedding, and index building
- `/docs/`: Directory for storing PDF documents (not present in the repository but referenced in code)
- `/cache/`: Directory for storing processed data and indexes (not present in the repository but created during execution)
  - `faiss_index.bin`: FAISS vector index file for semantic search
  - `faiss_metadata.json`: Metadata mapping index IDs to chunk information
  - `bm25_index.pkl`: Serialized BM25 index for keyword search
  - `chunks.json`: Processed document chunks

## Observed Patterns
- **Phase-based Development**: The project is clearly divided into two sequential phases:
  1. **Phase-1**: Document ingestion, chunking, embedding, and index building
  2. **Phase-2**: Query processing, retrieval, answer generation, confidence scoring, and caching
- **Modular Design**: Each phase has its own self-contained script with clear responsibilities
- **Configuration Through Constants**: Both scripts use global constants at the top for configuration
- **Environment Variables**: API keys and service configurations are loaded from `.env` files
- **Hybrid Retrieval Approach**: Combines semantic search (FAISS) and keyword search (BM25)
- **Caching Layer**: Redis is used for caching similar queries to reduce latency and API costs

## Details
The project follows a clear, sequential processing pipeline:

1. **Phase-1 (Document Processing)**:
   - Extracts text from PDF documents
   - Chunks text into paragraph-level segments
   - Generates embeddings using Gemini API
   - Builds hybrid search indexes (FAISS for semantic search, BM25 for keyword search)
   - Saves processed data and indexes to `/cache/` directory

2. **Phase-2 (Query Processing)**:
   - Classifies user queries (Irrelevant, General Q&A, or Company-Specific)
   - Retrieves relevant context from internal documents using hybrid search
   - Generates answers using Gemini LLM
   - Falls back to internet search when internal documents are insufficient
   - Calculates confidence scores for generated answers
   - Implements Redis-based caching for similar queries

The folder structure is minimal but effective, separating the two main phases of the RAG pipeline while keeping configuration and data in standard locations. The absence of more granular module separation suggests this is a focused, single-purpose application rather than a larger system with multiple components.