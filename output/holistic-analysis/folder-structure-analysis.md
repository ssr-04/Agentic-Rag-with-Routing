# Folder Structure Analysis Report

## Summary
The project is an Agentic Retrieval-Augmented Generation (RAG) system organized into two distinct phases. The codebase follows a clear phase-based architecture with separate Python scripts for each phase. Phase 1 focuses on data processing and indexing, while Phase 2 implements the intelligent agent for query handling. The folder structure is minimal and follows a logical progression from data ingestion to agent implementation.

## Directory Layout
- `/`: Root directory containing the main executable scripts and README
  - `Phase-1/`: Directory containing the implementation for the data processing and indexing phase
    - `phase1_build.py`: Script for building the hybrid index from document content
  - `agentic_rag_phase2.py`: Main script implementing the agentic RAG system
  - `README.md`: Documentation file explaining the system architecture and usage

## Expected Runtime Directories (not in repository)
- `cache/`: Directory created during runtime to store indices and processed data
  - Expected files: `faiss_index.bin`, `faiss_metadata.json`, `bm25_index.pkl`, `chunks.json`, `embeddings.json`
- `docs/`: Directory expected to contain PDF documents to be processed

## Observed Patterns
- **Phase-Based Structure**: The codebase is organized into distinct phases, with Phase 1 handling data processing and Phase 2 handling the agent logic.
- **Flat File Organization**: The project uses a flat organization with minimal nesting, focusing on functional separation rather than complex hierarchies.
- **Runtime-Generated Directories**: The system is designed to create necessary directories (`cache/`) during execution rather than having them pre-defined in the repository.
- **Clear Separation of Concerns**: Each script has a well-defined purpose - data processing in Phase 1 and agent logic in Phase 2.

## Details

### Phase 1: Data Processing and Indexing
The `Phase-1/phase1_build.py` script handles:
- PDF document extraction and cleaning
- Paragraph-level chunking of text
- Building hybrid search indices (FAISS for dense retrieval and BM25 for sparse retrieval)
- Creating and storing necessary metadata

### Phase 2: Agentic RAG Core Logic
The `agentic_rag_phase2.py` script implements:
- Query classification (Irrelevant, General Q&A, or Company-Specific)
- Hybrid retrieval from internal document indices
- LLM integration for answering and summarization
- Internet search fallback when needed
- Confidence scoring for generated answers
- Redis-based caching layer for similar queries

The folder structure reflects a modular, pipeline-based approach where data flows from document ingestion (Phase 1) to intelligent query handling (Phase 2). This organization facilitates clear separation of concerns and enables independent development and testing of each phase.