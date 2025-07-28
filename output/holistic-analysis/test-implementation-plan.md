# Test Implementation Plan for Agentic RAG System

## Overview

This document provides a practical implementation plan for adding tests to the Agentic RAG system. It includes ready-to-use test file templates, example test cases, and a step-by-step approach to building test coverage incrementally.

## Getting Started with Testing

### 1. Setting Up the Test Environment

First, create a testing directory structure and install the necessary dependencies:

```
f67dd938-4c96-41b1-a04b-cc918efa7ec6/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared test fixtures
│   ├── test_phase1.py           # Tests for phase1_build.py
│   ├── test_phase2.py           # Tests for agentic_rag_phase2.py
│   ├── test_integration.py      # End-to-end tests
│   ├── test_rag_metrics.py      # RAG-specific metrics
│   └── test_data/               # Test documents and fixtures
│       ├── test_docs/           # Test PDF documents
│       └── test_cache/          # Pre-built test indexes
```

Required test dependencies:
```
pytest
pytest-mock
coverage
rouge
bert-score
nltk
numpy
```

### 2. Create Shared Test Fixtures

Create a `conftest.py` file with shared fixtures:

```python
# tests/conftest.py
import os
import pytest
import tempfile
import shutil
import faiss
import pickle
import json
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def test_env():
    """Create a temporary test environment with controlled paths"""
    original_cache_dir = os.environ.get("CACHE_DIR", "cache")
    original_pdf_folder = os.environ.get("PDF_FOLDER", "docs")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_cache_dir = os.path.join(temp_dir, "cache")
        test_pdf_folder = os.path.join(temp_dir, "docs")
        os.makedirs(test_cache_dir)
        os.makedirs(test_pdf_folder)
        
        # Set environment variables for the test
        os.environ["CACHE_DIR"] = test_cache_dir
        os.environ["PDF_FOLDER"] = test_pdf_folder
        
        yield {
            "temp_dir": temp_dir,
            "cache_dir": test_cache_dir,
            "pdf_folder": test_pdf_folder
        }
        
        # Restore original environment
        os.environ["CACHE_DIR"] = original_cache_dir
        os.environ["PDF_FOLDER"] = original_pdf_folder

@pytest.fixture
def mock_gemini_embeddings():
    """Mock for Gemini embedding function"""
    def mock_embeddings(texts, task_type):
        # Return deterministic fake embeddings based on text content
        return [
            [float(hash(text) % 10000) / 10000 for _ in range(768)]
            for text in texts
        ]
    return mock_embeddings

@pytest.fixture
def test_faiss_index():
    """Create a small test FAISS index"""
    dimension = 768
    index = faiss.IndexFlatIP(dimension)
    # Add 5 test vectors
    vectors = np.random.random((5, dimension)).astype('float32')
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add(vectors)
    return index

@pytest.fixture
def test_faiss_metadata():
    """Create test metadata for FAISS index"""
    return {
        "0": {
            "chunk_id": "test_doc_c0",
            "source": "test_doc.pdf",
            "page_number": 1,
            "content": "The company vacation policy provides 20 days of paid time off per year.",
            "block_type": "paragraph",
            "parent": None
        },
        "1": {
            "chunk_id": "test_doc_c1",
            "source": "test_doc.pdf",
            "page_number": 1,
            "content": "Employees can carry over up to 5 unused vacation days to the next year.",
            "block_type": "paragraph",
            "parent": None
        },
        "2": {
            "chunk_id": "test_doc_c2",
            "source": "test_doc.pdf",
            "page_number": 2,
            "content": "The company offers flexible working hours between 7AM and 7PM.",
            "block_type": "paragraph",
            "parent": None
        },
        "3": {
            "chunk_id": "test_doc_c3",
            "source": "test_doc.pdf",
            "page_number": 3,
            "content": "Health insurance benefits include medical, dental, and vision coverage.",
            "block_type": "paragraph",
            "parent": None
        },
        "4": {
            "chunk_id": "test_doc_c4",
            "source": "test_doc.pdf",
            "page_number": 4,
            "content": "The 401k plan includes a 4% company match on employee contributions.",
            "block_type": "paragraph",
            "parent": None
        }
    }

@pytest.fixture
def test_bm25_index():
    """Create a simple BM25 index for testing"""
    from rank_bm25 import BM25Okapi
    corpus = [
        ["company", "vacation", "policy", "provides", "days", "paid", "time", "year"],
        ["employees", "carry", "unused", "vacation", "days", "next", "year"],
        ["company", "offers", "flexible", "working", "hours"],
        ["health", "insurance", "benefits", "include", "medical", "dental", "vision", "coverage"],
        ["401k", "plan", "includes", "company", "match", "employee", "contributions"]
    ]
    return BM25Okapi(corpus)

@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing cache operations"""
    mock_client = MagicMock()
    
    # Mock the json method and its set method
    mock_json = MagicMock()
    mock_client.json.return_value = mock_json
    
    # Mock the ft method and search method
    mock_ft = MagicMock()
    mock_client.ft.return_value = mock_ft
    
    # Storage for simulating cache
    mock_client._cache_storage = {}
    
    # Mock Redis search results
    class MockSearchResult:
        def __init__(self, total, docs):
            self.total = total
            self.docs = docs
    
    # Implement simplified json().set behavior
    def mock_set(key, path, value):
        mock_client._cache_storage[key] = value
        return True
    
    # Implement simplified ft().search behavior
    def mock_search(query, params):
        # Very simple simulation - just check if vector exists
        if params.get("vec") is not None:
            # Return empty result for testing
            return MockSearchResult(0, [])
        return MockSearchResult(0, [])
    
    mock_json.set.side_effect = mock_set
    mock_ft.search.side_effect = mock_search
    
    return mock_client
```

### 3. Implement Basic Unit Tests for Phase 1

Create `test_phase1.py` with initial tests for the document processing pipeline:

```python
# tests/test_phase1.py
import pytest
import os
from unittest.mock import patch, MagicMock
import sys
import json

# Add the project root to path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Phase_1.phase1_build import clean_text, extract_paragraphs, hierarchical_chunking

def test_clean_text():
    """Test text cleaning function"""
    # Test page number removal
    assert "This is text" == clean_text("This is text Page 5")
    
    # Test multiple newlines
    assert "Line1\nLine2" == clean_text("Line1\n\n\nLine2")
    
    # Test number-only line removal
    assert "Text before\nText after" == clean_text("Text before\n42\nText after")
    
    # Test combined cleaning
    dirty_text = "Page 10\n\n1234\n\nActual content\n\n\nMore content"
    expected = "Actual content\nMore content"
    assert expected == clean_text(dirty_text)

@patch('fitz.open')
def test_extract_paragraphs(mock_fitz_open):
    """Test PDF paragraph extraction with mocked PyMuPDF"""
    # Setup mock document
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Paragraph 1\n\nParagraph 2\n\nPage 5\n\n42"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz_open.return_value = mock_doc
    
    # Run extraction
    result = extract_paragraphs("fake_doc.pdf")
    
    # Verify results
    assert len(result) == 2
    assert result[0]["text"] == "Paragraph 1"
    assert result[0]["source"] == "fake_doc.pdf"
    assert result[0]["page_number"] == 1
    assert result[1]["text"] == "Paragraph 2"

def test_hierarchical_chunking():
    """Test the hierarchical chunking algorithm"""
    # Test with a short paragraph (should remain intact)
    blocks = [{
        "text": "This is a short paragraph.",
        "source": "test.pdf",
        "page_number": 1,
        "block_type": "paragraph"
    }]
    
    chunks = hierarchical_chunking(blocks)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "This is a short paragraph."
    assert chunks[0]["chunk_id"] == "test.pdf_c0"
    
    # Test with a long paragraph that needs splitting
    import sys
    # Save original MAX_TOKENS_PER_CHUNK
    from Phase_1.phase1_build import MAX_TOKENS_PER_CHUNK
    original_max = MAX_TOKENS_PER_CHUNK
    
    # Temporarily set MAX_TOKENS_PER_CHUNK to a small value for testing
    sys.modules["Phase_1.phase1_build"].MAX_TOKENS_PER_CHUNK = 5
    
    blocks = [{
        "text": "This is a long paragraph. It has multiple sentences. Each should be split correctly.",
        "source": "test.pdf",
        "page_number": 1,
        "block_type": "paragraph"
    }]
    
    chunks = hierarchical_chunking(blocks)
    assert len(chunks) > 1
    assert "This is a long paragraph." in chunks[0]["content"]
    
    # Restore original MAX_TOKENS_PER_CHUNK
    sys.modules["Phase_1.phase1_build"].MAX_TOKENS_PER_CHUNK = original_max
```

### 4. Implement Unit Tests for Phase 2

Create `test_phase2.py` for testing the core RAG functionality:

```python
# tests/test_phase2.py
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np
import json

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agentic_rag_phase2

def test_get_gemini_response(monkeypatch):
    """Test the Gemini response wrapper function"""
    # Mock the Gemini GenerativeModel
    mock_model = MagicMock()
    mock_generate = MagicMock()
    
    # Create a mock response structure that matches what the function expects
    mock_candidate = MagicMock()
    mock_content = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Test response"
    mock_content.parts = [mock_part]
    mock_candidate.content = mock_content
    mock_generate.return_value.candidates = [mock_candidate]
    
    mock_model.generate_content = mock_generate
    
    # Patch the GenerativeModel constructor to return our mock
    monkeypatch.setattr(agentic_rag_phase2.genai, 'GenerativeModel', lambda *args, **kwargs: mock_model)
    
    # Test the function
    result = agentic_rag_phase2.get_gemini_response("Test prompt")
    
    # Verify results
    assert result == "Test response"
    mock_model.generate_content.assert_called_once()

def test_classify_query(monkeypatch):
    """Test query classification with mocked LLM responses"""
    # Mock get_gemini_response to return controlled values
    responses = {
        "What is the capital of France?": agentic_rag_phase2.DECISION_IRRELEVANT,
        "How many vacation days do I get?": agentic_rag_phase2.DECISION_COMPANY_SPECIFIC,
        "What are good productivity tips?": agentic_rag_phase2.DECISION_GENERAL_QA
    }
    
    def mock_response(prompt, **kwargs):
        for query, response in responses.items():
            if query in prompt:
                return response
        return agentic_rag_phase2.DECISION_IRRELEVANT
    
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_response', mock_response)
    
    # Test the function with different queries
    assert agentic_rag_phase2.classify_query("What is the capital of France?") == agentic_rag_phase2.DECISION_IRRELEVANT
    assert agentic_rag_phase2.classify_query("How many vacation days do I get?") == agentic_rag_phase2.DECISION_COMPANY_SPECIFIC
    assert agentic_rag_phase2.classify_query("What are good productivity tips?") == agentic_rag_phase2.DECISION_GENERAL_QA

def test_hybrid_retrieval(test_faiss_index, test_faiss_metadata, test_bm25_index, monkeypatch):
    """Test hybrid retrieval functionality"""
    # Mock the embedding function
    def mock_embeddings(texts, task_type):
        # Return a specific embedding that should match our test vectors
        return [np.ones(768).tolist()]
    
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_embeddings', mock_embeddings)
    
    # Test retrieval
    results = agentic_rag_phase2.hybrid_retrieval(
        query="vacation policy",
        faiss_index=test_faiss_index,
        faiss_metadata=test_faiss_metadata,
        bm25_index=test_bm25_index,
        top_k=3,
        alpha=0.7
    )
    
    # Verify we got results
    assert len(results) > 0
    
    # Check that results have expected fields
    for result in results:
        assert "chunk_id" in result
        assert "source" in result
        assert "content" in result
        assert "score" in result

def test_evaluate_and_answer(monkeypatch):
    """Test the answer generation from context chunks"""
    # Test chunks
    test_chunks = [
        {
            "chunk_id": "test_doc_c0",
            "source": "test_doc.pdf",
            "page_number": 1,
            "content": "The company vacation policy provides 20 days of paid time off per year."
        }
    ]
    
    # Mock responses for different scenarios
    def mock_gemini_response(prompt, **kwargs):
        if "vacation" in prompt:
            return "Employees get 20 days of vacation per year. (Source: test_doc.pdf, Page 1)"
        elif "no context" in prompt:
            return agentic_rag_phase2.NO_ANSWER_IN_CONTEXT_SIGNAL
        else:
            return "Generic response"
    
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_response', mock_gemini_response)
    
    # Test with valid context
    answer, sources = agentic_rag_phase2.evaluate_and_answer("How many vacation days do I get?", test_chunks)
    assert "20 days" in answer
    assert len(sources) > 0
    
    # Test with irrelevant context
    answer, sources = agentic_rag_phase2.evaluate_and_answer("What is the meaning of life?", test_chunks)
    assert answer == agentic_rag_phase2.NO_ANSWER_IN_CONTEXT_SIGNAL
    assert len(sources) == 0

def test_confidence_score(monkeypatch):
    """Test confidence score calculation"""
    # Mock embeddings
    query_embedding = np.ones(768)
    
    # Test chunks with scores
    test_chunks = [
        {
            "chunk_id": "test_doc_c0",
            "source": "test_doc.pdf",
            "page_number": 1,
            "content": "The company vacation policy provides 20 days of paid time off per year.",
            "score": 0.9
        }
    ]
    
    # Mock embedding generation
    def mock_embeddings(texts, task_type):
        return [np.ones(768).tolist()]
    
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_embeddings', mock_embeddings)
    
    # Test RAG path
    confidence = agentic_rag_phase2.calculate_confidence_score(
        query_embedding=query_embedding,
        answer_text="Employees get 20 days of vacation per year.",
        retrieved_chunks=test_chunks,
        path_taken="Company_Specific_RAG_Success",
        raw_query="How many vacation days do I get?"
    )
    
    # Confidence should be between 0 and 1
    assert 0 <= confidence <= 1
    
    # Test Internet path
    confidence = agentic_rag_phase2.calculate_confidence_score(
        query_embedding=query_embedding,
        answer_text="Employees typically get 10-20 days of vacation per year in the US.",
        retrieved_chunks=[],
        path_taken="Internet_Search_Success",
        raw_query="How many vacation days do people get?"
    )
    
    assert 0 <= confidence <= 1
```

### 5. Implement RAG-Specific Metrics Tests

Create `test_rag_metrics.py` for specialized RAG evaluation metrics:

```python
# tests/test_rag_metrics.py
import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agentic_rag_phase2

# Define a test corpus with ground truth
test_corpus = [
    {
        "query": "What is the company's vacation policy?",
        "relevant_chunks": ["test_doc_c0", "test_doc_c1"],
        "gold_answer": "The company provides 20 days of vacation annually. Employees can carry over up to 5 unused days.",
        "classification": "COMPANY_SPECIFIC"
    },
    {
        "query": "What are the working hours?",
        "relevant_chunks": ["test_doc_c2"],
        "gold_answer": "The company offers flexible working hours between 7AM and 7PM.",
        "classification": "COMPANY_SPECIFIC"
    },
    {
        "query": "What health benefits are offered?",
        "relevant_chunks": ["test_doc_c3"],
        "gold_answer": "Health insurance benefits include medical, dental, and vision coverage.",
        "classification": "COMPANY_SPECIFIC"
    }
]

def test_retrieval_precision_recall(test_faiss_index, test_faiss_metadata, test_bm25_index, monkeypatch):
    """Test retrieval precision and recall metrics"""
    # Mock embeddings to return controlled values
    embeddings_map = {
        "vacation policy": [0.9, 0.1, 0.0, 0.0, 0.0] + [0.0] * 763,
        "working hours": [0.0, 0.0, 0.9, 0.0, 0.0] + [0.0] * 763,
        "health benefits": [0.0, 0.0, 0.0, 0.9, 0.0] + [0.0] * 763
    }
    
    def mock_embeddings(texts, task_type):
        for query, embedding in embeddings_map.items():
            if query in texts[0].lower():
                return [embedding]
        return [np.ones(768).tolist()]
    
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_embeddings', mock_embeddings)
    
    # Calculate metrics for each test case
    results = {}
    
    for test_case in test_corpus:
        query = test_case["query"]
        relevant_chunks = set(test_case["relevant_chunks"])
        
        # Get system results
        retrieved_chunks = agentic_rag_phase2.hybrid_retrieval(
            query=query,
            faiss_index=test_faiss_index,
            faiss_metadata=test_faiss_metadata,
            bm25_index=test_bm25_index,
            top_k=3
        )
        
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        
        # Calculate metrics
        true_positives = len(relevant_chunks.intersection(retrieved_ids))
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        recall = true_positives / len(relevant_chunks) if relevant_chunks else 0
        
        results[query] = {
            "precision": precision,
            "recall": recall,
            "retrieved": retrieved_ids,
            "relevant": list(relevant_chunks)
        }
    
    # Calculate average metrics
    avg_precision = sum(r["precision"] for r in results.values()) / len(results)
    avg_recall = sum(r["recall"] for r in results.values()) / len(results)
    
    # Print metrics for inspection
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print("Per-query results:")
    for query, metrics in results.items():
        print(f"  {query}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}")
        print(f"    Retrieved: {metrics['retrieved']}")
        print(f"    Relevant: {metrics['relevant']}")
    
    # We don't assert specific values since this is more for measurement
    # But we can verify the calculations work
    assert 0 <= avg_precision <= 1
    assert 0 <= avg_recall <= 1

def test_answer_quality(monkeypatch):
    """Test answer quality metrics"""
    try:
        from rouge import Rouge
    except ImportError:
        pytest.skip("Rouge library not available")
    
    # Mock the answer generation to return controlled answers
    def mock_answer_query(query, *args, **kwargs):
        if "vacation" in query.lower():
            return {
                "answer": "The company provides 20 days of vacation annually. Unused days can be carried over.",
                "path_taken": "Company_Specific_RAG_Success",
                "sources": []
            }
        elif "working hours" in query.lower():
            return {
                "answer": "The company has flexible working hours from 7AM to 7PM.",
                "path_taken": "Company_Specific_RAG_Success",
                "sources": []
            }
        else:
            return {
                "answer": "I don't have enough information to answer that question.",
                "path_taken": "No_Answer_Found",
                "sources": []
            }
    
    monkeypatch.setattr(agentic_rag_phase2, 'answer_query_agentic', mock_answer_query)
    
    # Initialize Rouge
    rouge = Rouge()
    
    results = {}
    for test_case in test_corpus:
        query = test_case["query"]
        gold_answer = test_case["gold_answer"]
        
        # Get system answer
        response = agentic_rag_phase2.answer_query_agentic(
            query, None, None, None  # Arguments don't matter due to mocking
        )
        generated_answer = response["answer"]
        
        # Calculate Rouge scores
        try:
            rouge_scores = rouge.get_scores(generated_answer, gold_answer)[0]
            
            results[query] = {
                "generated": generated_answer,
                "gold": gold_answer,
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"]
            }
        except ValueError:
            # Rouge can fail on very short texts
            results[query] = {
                "generated": generated_answer,
                "gold": gold_answer,
                "error": "Rouge calculation failed"
            }
    
    # Print results for inspection
    print("Answer Quality Results:")
    for query, metrics in results.items():
        print(f"  Query: {query}")
        print(f"    Generated: {metrics['generated']}")
        print(f"    Gold: {metrics['gold']}")
        if "rouge-1" in metrics:
            print(f"    ROUGE-1: {metrics['rouge-1']:.4f}")
            print(f"    ROUGE-L: {metrics['rouge-l']:.4f}")
        else:
            print(f"    Error: {metrics['error']}")
    
    # We don't assert specific values since this is more for measurement
```

### 6. Implement Integration Tests

Create `test_integration.py` for end-to-end tests:

```python
# tests/test_integration.py
import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import json

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agentic_rag_phase2
from Phase_1.phase1_build import build_pipeline

@pytest.mark.integration
def test_end_to_end_with_mocks(test_env, monkeypatch):
    """Test the end-to-end flow with mocked external dependencies"""
    # Mock PDF extraction to return controlled content
    mock_paragraphs = [
        {
            "source": "test_doc.pdf",
            "page_number": 1,
            "text": "The company vacation policy provides 20 days of paid time off per year.",
            "block_type": "paragraph"
        },
        {
            "source": "test_doc.pdf",
            "page_number": 1,
            "text": "Employees can carry over up to 5 unused vacation days to the next year.",
            "block_type": "paragraph"
        }
    ]
    
    def mock_extract_paragraphs(pdf_path):
        return mock_paragraphs
    
    # Mock embedding generation
    def mock_embeddings(texts, task_type):
        # Return deterministic embeddings
        return [[0.1 * i for i in range(768)] for _ in texts]
    
    # Apply mocks
    monkeypatch.setattr('Phase_1.phase1_build.extract_paragraphs', mock_extract_paragraphs)
    monkeypatch.setattr('Phase_1.phase1_build.get_gemini_embeddings_batch', mock_embeddings)
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_embeddings', lambda texts, task_type: [mock_embeddings([t], task_type)[0] for t in texts])
    
    # Create a test PDF file
    test_pdf_path = os.path.join(test_env["pdf_folder"], "test_doc.pdf")
    with open(test_pdf_path, 'wb') as f:
        f.write(b'%PDF-1.5\n%%EOF')  # Minimal PDF structure
    
    # Mock LLM responses
    def mock_gemini_response(prompt, **kwargs):
        if "classify" in prompt.lower():
            return agentic_rag_phase2.DECISION_COMPANY_SPECIFIC
        elif "answer" in prompt.lower() and "vacation" in prompt.lower():
            return "Employees get 20 days of vacation per year. (Source: test_doc.pdf, Page 1)"
        else:
            return "I don't have enough information to answer that question."
    
    monkeypatch.setattr(agentic_rag_phase2, 'get_gemini_response', mock_gemini_response)
    
    # Set up mock Redis cache
    mock_redis = MagicMock()
    mock_redis.json.return_value.set.return_value = True
    
    monkeypatch.setattr(agentic_rag_phase2.RedisCacheManager, '__init__', lambda self, **kwargs: None)
    monkeypatch.setattr(agentic_rag_phase2.RedisCacheManager, 'client', mock_redis)
    monkeypatch.setattr(agentic_rag_phase2.RedisCacheManager, 'retrieve_from_cache', lambda self, *args, **kwargs: None)
    monkeypatch.setattr(agentic_rag_phase2.RedisCacheManager, 'store_in_cache', lambda self, *args, **kwargs: None)
    
    # Run Phase 1 pipeline
    try:
        with patch('Phase_1.phase1_build.logger'):  # Silence logger
            build_pipeline()
    except Exception as e:
        pytest.fail(f"Phase 1 pipeline failed: {e}")
    
    # Verify Phase 1 artifacts
    assert os.path.exists(os.path.join(test_env["cache_dir"], "faiss_index.bin"))
    assert os.path.exists(os.path.join(test_env["cache_dir"], "faiss_metadata.json"))
    assert os.path.exists(os.path.join(test_env["cache_dir"], "bm25_index.pkl"))
    
    # Set cache paths for Phase 2
    agentic_rag_phase2.CACHE_DIR = test_env["cache_dir"]
    agentic_rag_phase2.FAISS_INDEX_FILE = os.path.join(test_env["cache_dir"], "faiss_index.bin")
    agentic_rag_phase2.FAISS_METADATA_FILE = os.path.join(test_env["cache_dir"], "faiss_metadata.json")
    agentic_rag_phase2.BM25_INDEX_FILE = os.path.join(test_env["cache_dir"], "bm25_index.pkl")
    
    # Load Phase 1 artifacts
    faiss_index, faiss_metadata = agentic_rag_phase2.load_faiss_index_and_metadata()
    bm25_index = agentic_rag_phase2.load_bm25_index()
    
    # Create cache manager
    cache_manager = agentic_rag_phase2.RedisCacheManager()
    
    # Test a query through Phase 2
    test_query = "How many vacation days do I get?"
    result = agentic_rag_phase2.answer_query_agentic_with_cache(
        test_query,
        faiss_index,
        faiss_metadata,
        bm25_index,
        cache_manager
    )
    
    # Verify results
    assert result is not None
    assert "answer" in result
    assert "20 days" in result["answer"]
    assert result["path_taken"] == "Company_Specific_RAG_Success"
    assert "confidence_score" in result
```

### 7. Create a Test Runner Script

Create a simple script to run all tests and generate a coverage report:

```python
# run_tests.py
#!/usr/bin/env python3
import os
import sys
import subprocess

def run_tests():
    """Run all tests and generate coverage report"""
    print("Running tests with coverage...")
    
    # Create the tests directory if it doesn't exist
    os.makedirs("tests", exist_ok=True)
    
    # Create an empty __init__.py file if it doesn't exist
    init_file = os.path.join("tests", "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            pass
    
    # Run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--cov=Phase_1",
        "--cov=agentic_rag_phase2",
        "--cov-report=term",
        "--cov-report=html:coverage_report"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\nTests passed successfully!")
        print("Coverage report generated in 'coverage_report' directory")
    else:
        print("\nSome tests failed.")
        sys.exit(result.returncode)

if __name__ == "__main__":
    run_tests()
```

## Implementation Plan

Follow this step-by-step plan to implement testing for the RAG system:

1. **Set up the testing framework**:
   ```bash
   mkdir -p tests/test_data
   touch tests/__init__.py
   pip install pytest pytest-mock coverage
   ```

2. **Create the basic test fixtures**:
   - Copy the `conftest.py` content from this document
   - Adjust paths and imports as needed for your project structure

3. **Start with simple unit tests**:
   - Implement `test_phase1.py` focusing on text processing functions
   - Run with `pytest tests/test_phase1.py -v`

4. **Add Phase 2 unit tests**:
   - Implement `test_phase2.py` focusing on core RAG functions
   - Run with `pytest tests/test_phase2.py -v`

5. **Implement specialized RAG metrics**:
   - Add `test_rag_metrics.py` for retrieval and generation quality
   - Run with `pytest tests/test_rag_metrics.py -v`

6. **Add integration tests**:
   - Implement `test_integration.py` for end-to-end testing
   - Run with `pytest tests/test_integration.py -v`

7. **Generate coverage reports**:
   - Run all tests with coverage: `pytest --cov=. tests/`
   - Identify areas with low coverage and add tests

8. **Automate testing**:
   - Use the `run_tests.py` script to run all tests and generate reports

## Conclusion

This test implementation plan provides a comprehensive approach to testing the Agentic RAG system. By starting with simple unit tests and gradually adding more complex tests for specialized RAG functionality, you can build a robust test suite that ensures the reliability and accuracy of the system.

Key recommendations:
1. Focus first on testing the core algorithms with unit tests
2. Use mocking extensively to isolate components from external dependencies
3. Create specialized tests for RAG-specific metrics
4. Build a comprehensive test dataset with ground truth
5. Implement integration tests for end-to-end validation

By following this plan, you'll establish a strong testing foundation that can evolve with the system and provide confidence in its behavior across various scenarios.