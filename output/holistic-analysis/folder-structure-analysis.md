# Folder Structure Analysis Report

## Summary
This repository implements a two-phase Agentic Retrieval-Augmented Generation (RAG) system. The codebase follows a simple, phase-based organization with clear separation between the document processing pipeline (Phase-1) and the intelligent query processing system (Phase-2). The structure is minimal but functional, with each phase having dedicated scripts that handle specific parts of the RAG workflow.

## Directory Layout
- `/`: Root directory containing the main Phase 2 implementation (`agentic_rag_phase2.py`) and README
- `/Phase-1`: Contains the implementation for document processing and index building (`phase1_build.py`)
- `/docs`: Not explicitly visible in the folder structure, but referenced in the code as the location for PDF documents to be processed
- `/cache`: Not explicitly visible in the folder structure, but referenced in code as the storage location for processed data, embeddings, and indexes

## Observed Patterns
- **Phase-based Architecture**: The codebase is organized into distinct phases that represent the RAG pipeline workflow:
  - Phase 1: Document ingestion, chunking, embedding, and index building
  - Phase 2: Query processing, retrieval, answer generation, and caching
  
- **Flat Structure**: The repository uses a flat structure with minimal nesting, suggesting a focus on simplicity and functionality over complex organizational patterns.

- **Implicit Directories**: Some directories (`docs`, `cache`) are referenced in code but not explicitly visible in the folder structure, suggesting they are created at runtime.

- **Standalone Scripts**: Each phase is implemented as a standalone Python script with a clear entry point and command-line interface.

## Details

### Phase-1 (High Confidence)
The Phase-1 directory contains `phase1_build.py` which implements the document processing pipeline. This script:
- Processes PDF documents from the `docs` folder
- Extracts and cleans text from PDFs
- Implements paragraph-level chunking
- Generates embeddings using Gemini's text-embedding model
- Builds hybrid search indexes (FAISS for dense retrieval and BM25 for sparse retrieval)
- Saves processed data to the `cache` directory

### Phase-2 (High Confidence)
The Phase-2 implementation is contained in the root directory as `agentic_rag_phase2.py`. This script:
- Loads the indexes and metadata created by Phase-1
- Implements an intelligent agent that classifies queries
- Performs hybrid retrieval from internal documents
- Generates answers using LLM with retrieved context
- Falls back to internet search when internal documents are insufficient
- Calculates confidence scores for generated answers
- Implements Redis-based caching for similar queries

### Runtime-Generated Directories (Medium Confidence)
Based on code references, the system creates or expects:
- A `docs` directory containing PDF documents to be processed
- A `cache` directory for storing processed data, embeddings, and indexes

### Development Approach (Medium Confidence)
The code organization suggests an iterative development approach, with Phase-1 focusing on document processing and indexing, and Phase-2 building upon those artifacts to implement the intelligent query-answering system. The clean separation between phases allows for independent development and testing of each component.