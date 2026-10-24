# Hybrid KG-RAG Knowledge Assistant

A knowledge-based question-answering application built with **Python, Streamlit, ChromaDB, NetworkX, Sentence Transformers, and OpenAI**.

This project demonstrates a **Hybrid Retrieval-Augmented Generation (RAG)** approach that combines vector similarity search with Knowledge Graph (KG) retrieval to answer questions from a PDF knowledge base.

## 🚀 Project Overview

Traditional RAG retrieves relevant text chunks using vector similarity. This project extends that approach by also extracting relationships between entities and storing them in a Knowledge Graph.

The application combines:

- **Vector Search:** Retrieves semantically relevant PDF chunks using embeddings.
- **Knowledge Graph:** Stores entities and their relationships using NetworkX.
- **Hybrid Retrieval:** Combines vector results with graph-related context.
- **LLM-Based Answers:** Uses OpenAI to generate answers grounded in retrieved passages.
- **Interactive UI:** Provides a Streamlit interface to process documents and ask questions.
- **Source References:** Displays supporting passages and page information to help verify answers.

## 🏗️ Architecture

```text
             PDF Knowledge Base
                     |
                     v
              PDF Text Extraction
                     |
                     v
                Text Chunking
                     |
          +----------+----------+
          |                     |
          v                     v
   Text Embeddings       Relationship Extraction
          |                     |
          v                     v
       ChromaDB             Knowledge Graph
    (Vector Database)        (NetworkX)
          |                     |
          +----------+----------+
                     |
                     v
               Hybrid Retrieval
                     |
                     v
                OpenAI LLM
                     |
                     v
             Answer with Sources
                     |
                     v
               Streamlit UI
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Interactive web interface |
| PyPDF | Extract
