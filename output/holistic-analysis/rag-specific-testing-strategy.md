# RAG-Specific Testing Strategy

## Summary
Testing a Retrieval-Augmented Generation (RAG) system presents unique challenges due to the combination of retrieval algorithms, large language models, and the subjective nature of generated answers. This document outlines specialized testing approaches for the Agentic RAG system that focus on evaluating both the retrieval accuracy and the quality of generated answers.

## RAG-Specific Testing Challenges

### 1. Retrieval Evaluation
- **Relevance Assessment**: Determining whether retrieved chunks are truly relevant to a query
- **Ranking Quality**: Evaluating if the most relevant chunks are ranked highest
- **Hybrid Algorithm Balance**: Testing the effectiveness of combining dense and sparse retrieval

### 2. Generation Evaluation
- **Answer Accuracy**: Assessing if generated answers are factually correct
- **Hallucination Detection**: Identifying when the LLM generates information not present in retrieved documents
- **Citation Accuracy**: Verifying that citations correctly reference source documents

### 3. System Behavior
- **Query Classification**: Testing if queries are correctly routed to appropriate processing paths
- **Confidence Scoring**: Validating that confidence scores correlate with answer quality
- **Caching Effectiveness**: Measuring the impact of caching on performance and accuracy

## Specialized Testing Approaches

### 1. Ground Truth Dataset Creation

**Recommendation**: Create a curated test dataset with known query-document-answer relationships.

**Implementation Steps**:
1. Select representative documents for the test corpus
2. Generate diverse queries relevant to these documents
3. Manually identify the ideal chunks that should be retrieved for each query
4. Create gold-standard answers that correctly use information from these chunks

**Example Structure**:
```python
test_corpus = [
    {
        "query": "What is the company's vacation policy?",
        "relevant_chunks": ["policy_doc_c12", "policy_doc_c15"],
        "gold_answer": "The company provides 20 days of vacation annually (Source: policy_doc, Page 5).",
        "classification": "COMPANY_SPECIFIC"
    },
    # More test cases...
]
```

### 2. Retrieval Metrics Testing

**Recommendation**: Implement standard information retrieval metrics to evaluate retrieval quality.

**Key Metrics to Implement**:
- **Precision@k**: Proportion of retrieved documents that are relevant
- **Recall@k**: Proportion of relevant documents that are retrieved
- **Mean Average Precision (MAP)**: Average precision across multiple queries
- **Normalized Discounted Cumulative Gain (nDCG)**: Measures ranking quality with relevance grades

**Example Implementation**:
```python
def test_retrieval_metrics():
    results = {}
    
    for test_case in test_corpus:
        query = test_case["query"]
        relevant_chunks = set(test_case["relevant_chunks"])
        
        # Get system results
        retrieved_chunks = hybrid_retrieval(
            query=query,
            faiss_index=test_faiss_index,
            faiss_metadata=test_metadata,
            bm25_index=test_bm25_index,
            top_k=5
        )
        
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        
        # Calculate metrics
        precision = len(relevant_chunks.intersection(retrieved_ids)) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(relevant_chunks.intersection(retrieved_ids)) / len(relevant_chunks) if relevant_chunks else 0
        
        results[query] = {
            "precision": precision,
            "recall": recall,
            # Calculate other metrics...
        }
    
    # Aggregate results
    avg_precision = sum(r["precision"] for r in results.values()) / len(results)
    avg_recall = sum(r["recall"] for r in results.values()) / len(results)
    
    assert avg_precision > 0.7, f"Average precision {avg_precision} below threshold"
    assert avg_recall > 0.5, f"Average recall {avg_recall} below threshold"
```

### 3. Answer Quality Evaluation

**Recommendation**: Implement automated metrics and human evaluation protocols for answer quality.

**Automated Metrics**:
- **BLEU/ROUGE/BERTScore**: Compare generated answers to gold standard answers
- **Entity Matching**: Check if key entities from gold answers appear in generated answers
- **Citation Count**: Verify that generated answers include an appropriate number of citations
- **Hallucination Detection**: Compare answer content to retrieved chunks to identify unsupported statements

**Human Evaluation Protocol**:
- Create a rubric for manual scoring (relevance, accuracy, completeness)
- Implement blind comparison tests between system versions
- Track subjective quality scores over time

**Example Implementation**:
```python
from rouge import Rouge
from bert_score import BERTScorer

def test_answer_quality():
    rouge = Rouge()
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    
    results = {}
    for test_case in test_corpus:
        query = test_case["query"]
        gold_answer = test_case["gold_answer"]
        
        # Get system answer
        response = answer_query_agentic(
            query,
            test_faiss_index,
            test_metadata,
            test_bm25_index
        )
        generated_answer = response["answer"]
        
        # Calculate metrics
        rouge_scores = rouge.get_scores(generated_answer, gold_answer)[0]
        _, _, bert_f1 = bert_scorer.score([generated_answer], [gold_answer])
        
        # Entity matching
        entities_present = check_entities_present(generated_answer, gold_answer)
        
        # Citation verification
        citations_present = len(re.findall(r"\(Source:.*?\)", generated_answer))
        expected_citations = len(re.findall(r"\(Source:.*?\)", gold_answer))
        
        results[query] = {
            "rouge_1_f": rouge_scores["rouge-1"]["f"],
            "rouge_l_f": rouge_scores["rouge-l"]["f"],
            "bert_score": bert_f1.item(),
            "entities_match": entities_present,
            "citation_ratio": citations_present / expected_citations if expected_citations else 0
        }
    
    # Aggregate and assert
    avg_rouge_1 = sum(r["rouge_1_f"] for r in results.values()) / len(results)
    assert avg_rouge_1 > 0.4, f"Average ROUGE-1 {avg_rouge_1} below threshold"
```

### 4. Hallucination Testing

**Recommendation**: Implement specific tests to detect and measure hallucinations in generated answers.

**Implementation Approaches**:
1. **Fact Extraction and Verification**:
   - Extract factual claims from generated answers
   - Check if each claim is supported by retrieved documents
   
2. **Controlled Misinformation**:
   - Intentionally include documents with conflicting information
   - Test if the system appropriately handles contradictions
   
3. **Out-of-Context Queries**:
   - Test with queries that have no relevant information in the corpus
   - Verify the system correctly indicates insufficient information

**Example Implementation**:
```python
def test_hallucination_detection():
    # Test cases designed to trigger hallucinations
    hallucination_test_cases = [
        {
            "query": "What is the company's policy on sabbaticals?",
            "in_corpus": False,  # No information about this in the corpus
            "expected_response": "NO_ANSWER_IN_CONTEXT_SIGNAL"
        },
        {
            "query": "How many vacation days do employees get?",
            "in_corpus": True,
            "contradictory_docs": True,  # Corpus has conflicting information
            "expected_behavior": "acknowledge_contradiction"
        }
    ]
    
    for test_case in hallucination_test_cases:
        response = answer_query_agentic(
            test_case["query"],
            test_faiss_index,
            test_metadata,
            test_bm25_index
        )
        
        if not test_case["in_corpus"]:
            # Should indicate no information available
            assert "NO_ANSWER_IN_CONTEXT" in response["path_taken"] or \
                   "I couldn't find information" in response["answer"]
```

### 5. Classification Testing

**Recommendation**: Test the query classification system with diverse query types.

**Test Categories**:
- Clear company-specific queries
- Ambiguous queries that could be either company-specific or general
- Clearly irrelevant queries
- Edge cases (very short queries, queries with typos)

**Example Implementation**:
```python
@pytest.mark.parametrize("query,expected_class,confidence", [
    # Clear cases
    ("What is the vacation policy?", "COMPANY_SPECIFIC", "high"),
    ("Who is the CEO of Google?", "IRRELEVANT", "high"),
    ("How do I improve productivity?", "GENERAL_QA", "high"),
    
    # Ambiguous cases
    ("What are working hours?", "COMPANY_SPECIFIC", "medium"),
    ("How do I request time off?", "COMPANY_SPECIFIC", "medium"),
    
    # Edge cases
    ("PTO?", "COMPANY_SPECIFIC", "low"),
    ("vacaton policy", "COMPANY_SPECIFIC", "medium"),  # Typo
])
def test_query_classification(query, expected_class, confidence, monkeypatch):
    # Mock get_gemini_response to return our expected class for testing
    monkeypatch.setattr(
        'agentic_rag_phase2.get_gemini_response',
        lambda *args, **kwargs: expected_class
    )
    
    result = classify_query(query)
    assert result == expected_class
```

### 6. Confidence Score Validation

**Recommendation**: Validate that confidence scores correlate with answer quality.

**Implementation Approach**:
1. Generate answers for a test set with known gold standard answers
2. Calculate confidence scores for each answer
3. Measure correlation between confidence scores and answer quality metrics
4. Test if high-confidence answers are indeed more accurate

**Example Implementation**:
```python
def test_confidence_score_correlation():
    results = []
    
    for test_case in test_corpus:
        query = test_case["query"]
        gold_answer = test_case["gold_answer"]
        
        # Get system answer with confidence
        query_embedding = get_gemini_embeddings([query], task_type="RETRIEVAL_QUERY")[0]
        response = answer_query_agentic(
            query,
            test_faiss_index,
            test_metadata,
            test_bm25_index
        )
        
        # Calculate answer quality metrics
        answer_quality = calculate_answer_quality(response["answer"], gold_answer)
        
        # Calculate confidence score
        confidence = calculate_confidence_score(
            np.array(query_embedding),
            response["answer"],
            response.get("sources", []),
            response["path_taken"],
            query
        )
        
        results.append({
            "query": query,
            "confidence": confidence,
            "quality": answer_quality
        })
    
    # Calculate correlation
    confidence_values = [r["confidence"] for r in results]
    quality_values = [r["quality"] for r in results]
    correlation = np.corrcoef(confidence_values, quality_values)[0, 1]
    
    assert correlation > 0.5, f"Confidence-quality correlation {correlation} below threshold"
```

### 7. Caching System Testing

**Recommendation**: Test the effectiveness and accuracy of the Redis caching system.

**Test Scenarios**:
- Exact query repetition
- Semantically similar queries
- Cache expiration/invalidation
- Performance impact

**Example Implementation**:
```python
def test_cache_hit_accuracy():
    cache_manager = RedisCacheManager(
        host="localhost",
        port=6379,
        db=0,
        index_name="test_rag_cache",
        vector_dimension=768
    )
    
    # Clear test cache
    cache_manager.client.flushdb()
    
    # Original query
    original_query = "What is the company's vacation policy?"
    original_embedding = get_gemini_embeddings([original_query], task_type="RETRIEVAL_QUERY")[0]
    
    # Generate and cache a response
    original_response = answer_query_agentic(
        original_query,
        test_faiss_index,
        test_metadata,
        test_bm25_index
    )
    original_response["confidence_score"] = 0.9  # High confidence for testing
    
    cache_manager.store_in_cache(
        original_query,
        np.array(original_embedding),
        original_response
    )
    
    # Test with semantically similar query
    similar_query = "Tell me about the vacation policy at the company"
    similar_embedding = get_gemini_embeddings([similar_query], task_type="RETRIEVAL_QUERY")[0]
    
    cached_response = cache_manager.retrieve_from_cache(
        np.array(similar_embedding),
        CACHING_SIMILARITY_THRESHOLD
    )
    
    assert cached_response is not None, "Cache miss for semantically similar query"
    assert cached_response["cached"] == True
    assert cached_response["cache_similarity"] >= CACHING_SIMILARITY_THRESHOLD
```

## Performance Testing for RAG

**Recommendation**: Implement specific performance tests for RAG components.

**Key Performance Metrics**:
- **Indexing Speed**: Time to process and index documents
- **Query Latency**: End-to-end response time
- **Retrieval Speed**: Time for vector search and hybrid scoring
- **Cache Hit Ratio**: Percentage of queries served from cache
- **Memory Usage**: RAM consumption during indexing and querying

**Example Implementation**:
```python
def test_retrieval_performance():
    import time
    
    queries = [
        "What is the vacation policy?",
        "How do I request time off?",
        "What are the working hours?",
        # Add more diverse queries...
    ]
    
    # Warm-up
    hybrid_retrieval(
        queries[0],
        test_faiss_index,
        test_metadata,
        test_bm25_index
    )
    
    # Measure retrieval times
    retrieval_times = []
    for query in queries:
        start_time = time.time()
        results = hybrid_retrieval(
            query,
            test_faiss_index,
            test_metadata,
            test_bm25_index,
            top_k=5
        )
        end_time = time.time()
        retrieval_times.append(end_time - start_time)
    
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    max_retrieval_time = max(retrieval_times)
    
    assert avg_retrieval_time < 0.5, f"Average retrieval time {avg_retrieval_time}s exceeds threshold"
    assert max_retrieval_time < 1.0, f"Maximum retrieval time {max_retrieval_time}s exceeds threshold"
```

## Integration with Continuous Improvement

**Recommendation**: Implement a testing framework that supports continuous improvement of the RAG system.

**Implementation Components**:
1. **Regression Test Suite**: Maintain a growing set of test queries and expected behaviors
2. **A/B Testing Framework**: Compare different retrieval algorithms or LLM configurations
3. **Feedback Loop Integration**: Capture user feedback to create new test cases
4. **Benchmark Version Control**: Track performance metrics across system versions

## Conclusion

Testing a RAG system requires specialized approaches that go beyond traditional software testing. By implementing the strategies outlined in this document, the team can systematically evaluate and improve both the retrieval and generation aspects of the system.

The most critical components to focus on initially are:
1. Creating a high-quality ground truth dataset
2. Implementing retrieval quality metrics
3. Developing hallucination detection tests

These foundations will enable more sophisticated testing approaches as the system evolves and provide confidence in the system's ability to retrieve relevant information and generate accurate answers.