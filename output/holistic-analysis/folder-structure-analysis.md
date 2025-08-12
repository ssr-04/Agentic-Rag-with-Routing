# Folder Structure Analysis Report

## Summary
This repository implements an Agentic Retrieval-Augmented Generation (RAG) system developed in two distinct phases. The codebase is organized in a simple, phase-based structure with clear separation between the data processing/indexing functionality (Phase 1) and the intelligent agent core logic (Phase 2). The project uses a minimalist folder structure with a focus on Python scripts rather than complex module hierarchies.

## Directory Layout
- `/`: Root directory containing the main Phase 2 script (`agentic_rag_phase2.py`), README, and Phase 1 subdirectory
- `/Phase-1`: Contains the data processing and indexing script (`phase1_build.py`) for preparing document data
- `/docs` (Referenced in code): Expected directory for source PDF documents (not present in the repository structure)
- `/cache` (Referenced in code): Expected directory for storing processed data, embeddings, and indexes (not present in the repository structure)

## Observed Patterns
- **Phase-Based Organization**: The codebase is clearly divided into two phases, with Phase 1 handling document processing and indexing, and Phase 2 implementing the intelligent agent logic.
- **Flat Structure**: The repository uses a flat structure with minimal nesting, suggesting a focus on simplicity and ease of use.
- **Functionality-Based Separation**: Rather than organizing by technical layers (e.g., models, views, controllers), the code is organized by functional phases in the RAG pipeline.
- **Runtime Directory Creation**: Both Phase 1 and Phase 2 scripts dynamically create necessary directories (`cache` and possibly `docs`) at runtime if they don't exist.
- **No Traditional Package Structure**: The codebase doesn't follow traditional Python package organization with `__init__.py` files and module imports, suggesting it's designed as standalone scripts rather than a reusable library.

## Details

### File Purposes
- `README.md`: Comprehensive documentation explaining the system's purpose, features, prerequisites, and usage instructions.
- `Phase-1/phase1_build.py`: Responsible for:
  - Processing PDF documents from a `docs` directory
  - Extracting and cleaning text from PDFs
  - Chunking text at paragraph level with intelligent boundary detection
  - Generating embeddings using Gemini API
  - Building hybrid search indexes (FAISS for vector search and BM25 for keyword search)
  - Storing processed data and indexes in the `cache` directory
- `agentic_rag_phase2.py`: Implements the intelligent agent that:
  - Classifies user queries
  - Retrieves relevant information from indexes created in Phase 1
  - Leverages LLM knowledge for general questions
  - Performs internet searches when necessary
  - Generates answers with confidence scores
  - Implements Redis-based caching for similar questions

### Runtime-Created Directories
- `cache/`: Stores processed data and indexes:
  - `faiss_index.bin`: FAISS vector index
  - `faiss_metadata.json`: Metadata mapping index IDs to chunk information
  - `bm25_index.pkl`: Serialized BM25 keyword index
  - `chunks.json`: Processed text chunks
  - `embeddings.json`: Generated embeddings
- `docs/`: Expected to contain source PDF documents

### Design Philosophy
The folder structure reflects a pragmatic, task-oriented approach focused on the RAG pipeline workflow rather than software architecture patterns. This suggests the project prioritizes functionality and results over architectural complexity, making it accessible for users who want to understand the RAG system implementation without navigating complex module relationships.

### Confidence Level: High
The folder structure analysis is based on direct examination of the repository files and code content. The purpose and relationships between files are clearly documented in both the README and code comments.