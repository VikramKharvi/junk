
# Multi-Model LLM System with Router

[](https://github.com/VikramKharvi/junk/tree/main#multi-model-llm-system-with-router)

A comprehensive system that uses two specialized LLMs trained on different topics and implements an intelligent router to direct queries to the appropriate model based on the input question.

## Project Overview

This system demonstrates a multi-model approach where:

-   Model A: Llama 3.1 8B specialized in car industry knowledge
-   Model B: Phi-3 Medium specialized in sleep science
-   Router: Sentence transformer-based classification system for query routing

## Architecture


1.  Model Training Pipeline
2.  Intelligent Router System
3.  Inference Pipeline
4.  Performance Evaluation Framework

## Requirements

-   Python 3.8+
-   CUDA-compatible GPU (Tesla T4 or better) or Apple M chip
-   16GB+ GPU memory recommended
-   PyTorch 2.6.0+
-   Transformers 4.55.0+

## Installation

# Install core dependencies
pip install unsloth
pip install sentence-transformers rouge-score scikit-learn
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
!pip install --no-deps unsloth

## Quick Start

[](https://github.com/VikramKharvi/junk/tree/main#quick-start)

1.  Prepare your datasets in JSON format:
    
    {
      "qna": [
        {"question": "Your question here", "answer": "Your answer here"}
      ]
    }
    
2.  Run the training notebook:
    
    jupyter notebook webai.ipynb
    
3.  Use the trained models:
    
    from your_module import process_query
    
    result = process_query("How do I change my car's oil?")
    print(result)
    

# System Components

[](https://github.com/VikramKharvi/junk/tree/main#system-components)

1.  Model Training

----------

-   **Car Model**: Llama 3.1 8B fine-tuned with LoRA on car industry Q&A
-   **Sleep Model**: Phi-3 Medium fine-tuned with LoRA on sleep science Q&A
-   **Training Approach**: Supervised Fine-Tuning (SFT) with PEFT/LoRA
-   **Optimization**: Unsloth for 2x faster training, gradient checkpointing

2.  Router Implementation

----------

-   **Classification Method**: Sentence transformer embeddings with cosine similarity
-   **Model**: all-MiniLM-L6-v2 for efficient text classification
-   **Categories**: "car" and "sleep" with confidence scoring
-   **Accuracy**: ~85% routing accuracy on test queries

3.  Inference Pipeline

----------

-   **Query Processing**: Automatic topic classification
-   **Model Selection**: Dynamic routing to appropriate specialized model
-   **Response Generation**: Context-aware answers with proper formatting
-   **Performance**: ~10 seconds per query on Tesla T4

4.  Performance Evaluation

----------

-   **Metrics**: ROUGE-1, ROUGE-2, ROUGE-L scores
-   **Datasets**: Separate validation sets for each domain
-   **Benchmarking**: Cross-domain performance comparison

# Key Features

[](https://github.com/VikramKharvi/junk/tree/main#key-features)

-   **Efficient Training**: LoRA adapters reduce trainable parameters by 99%
-   **Smart Routing**: NLP-based query classification for optimal model selection
-   **Memory Optimization**: 4-bit quantization and gradient checkpointing
-   **Scalable Architecture**: Easy to add new domains and models
-   **Performance Monitoring**: Built-in evaluation and benchmarking tools

# Design Choices & Assumptions

[](https://github.com/VikramKharvi/junk/tree/main#design-choices--assumptions)

1.  Model Selection

----------

-   **Llama 3.1 8B**: Chosen for car domain due to strong reasoning capabilities
-   **Phi-3 Medium**: Selected for sleep domain for efficiency and performance
-   **LoRA**: Used for parameter-efficient fine-tuning, reducing memory requirements

2.  Router Design

----------

-   **Sentence Transformers**: Chosen over traditional ML classifiers for better semantic understanding
-   **Cosine Similarity**: Simple but effective similarity metric for binary classification
-   **Two-Domain Assumption**: System assumes queries belong to exactly one domain

3.  Training Strategy

----------

-   **Small Datasets**: Optimized for scenarios with limited training data
-   **Quick Iteration**: 60 training steps for rapid prototyping
-   **Validation Split**: 80/20 train/validation split for reliable evaluation

# Challenges & Solutions

[](https://github.com/VikramKharvi/junk/tree/main#challenges--solutions)

1.  Memory Constraints

----------

-   **Challenge**: Large models require significant GPU memory
-   **Solution**: 4-bit quantization, LoRA adapters, gradient checkpointing

2.  Training Efficiency

----------

-   **Challenge**: Full fine-tuning is slow and resource-intensive
-   **Solution**: Unsloth optimization, LoRA for parameter-efficient training

3.  Router Accuracy

----------

-   **Challenge**: Ambiguous queries that could belong to multiple domains
-   **Solution**: Semantic similarity scoring with confidence thresholds

# Performance Metrics

[](https://github.com/VikramKharvi/junk/tree/main#performance-metrics)

-   **Routing Accuracy**: 85% on test queries
-   **Training Time**: ~2-3 minutes per model on Tesla T4
-   **Memory Usage**: 6-14 GB GPU memory during training
-   **Inference Speed**: ~10 seconds per query
-   **ROUGE Scores**: ROUGE-L ~0.66 for car domain, varies for sleep domain

# Timeline Options

[](https://github.com/VikramKharvi/junk/tree/main#timeline-options)

## Small Effort (2-3 weeks)

[](https://github.com/VikramKharvi/junk/tree/main#small-effort-2-3-weeks)

-   Basic two-model system with simple router
-   Hyperparameter tunning
-   Other robust Evaluation Mechanism
-   Essential training and inference pipeline
-   Basic documentation and evaluation
-   Current implementation level

## Medium Effort (4-6 weeks)

[](https://github.com/VikramKharvi/junk/tree/main#medium-effort-4-6-weeks)

-   Enhanced router with confidence scoring
-   Hot swap adapters
-   Try different embedding models
-   Better evaluation metrics and benchmarking
-   Web interface for easy testing
-   Model versioning and management

## Large Effort (8-12 weeks)

[](https://github.com/VikramKharvi/junk/tree/main#large-effort-8-12-weeks)

-   Multi-domain support (5+ domains)
-   Advanced routing with ensemble methods
-   Conversational interface with memory
-   Real-time performance monitoring
-   Model compression and optimization
-   Production deployment pipeline
-   Comprehensive testing suite

# Future Improvements

[](https://github.com/VikramKharvi/junk/tree/main#future-improvements)

1.  **Router Enhancement**
    
    -   Multi-label classification for overlapping domains
    -   Confidence-based routing with fallback models
    -   Dynamic threshold adjustment
2.  **Model Optimization**
    
    -   Knowledge distillation for smaller models
    -   Quantization-aware training
    -   Model pruning and compression
3.  **System Architecture**
    
    -   Load balancing for multiple model instances
    -   Caching layer for frequent queries
    -   A/B testing framework for model selection
4.  **Evaluation Framework**
    
    -   Human evaluation metrics
    -   Domain-specific quality measures
    -   Continuous learning from user feedback

# Usage Examples

[](https://github.com/VikramKharvi/junk/tree/main#usage-examples)

# Basic query processing
query = "What causes insomnia?"
result = process_query(query)
print(f"Answer:", result)

# Batch processing
queries = [
    "How do hybrid engines work?",
    "What are the stages of sleep?",
    "Best SUV for off-road driving?"
]

for query in queries:
	result = process_query(query)
	print(f"Answer:", result)
