# RAG Pipeline with LangChain

A complete implementation of Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and HuggingFace embeddings.

## Overview

This repository demonstrates how to build a RAG system from scratch, covering all essential components:
- Document loading from PDFs
- Text splitting and chunking
- Embedding generation
- Vector storage with ChromaDB
- Semantic search and retrieval
- LLM integration for question answering

## Features

- **Multiple Document Loaders**: PDF, text files, and web content
- **Flexible Text Splitting**: Configurable chunk sizes and overlap
- **Local Embeddings**: HuggingFace sentence transformers (no API key needed)
- **Vector Store**: ChromaDB for efficient similarity search
- **Advanced Retrieval**: Similarity search and MMR (Maximum Marginal Relevance)
- **LLM Integration**: Support for Groq, OpenAI, HuggingFace, and Ollama
- **Interactive Notebooks**: Step-by-step Jupyter notebooks

## Project Structure

```
.
├── noteboook/
│   ├── embeddings.ipynb          # Embedding generation and similarity
│   ├── chromadb.ipynb             # Complete RAG pipeline
│   ├── pdfdataparsing.ipynb       # PDF parsing examples
│   ├── Documents.ipynb            # Document loading
│   └── data/
│       ├── pdf/                   # PDF documents
│       └── text_fils/             # Text files
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project configuration
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-pipeline-langchain.git
cd rag-pipeline-langchain
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

1. Open the main RAG notebook:
```bash
jupyter notebook noteboook/chromadb.ipynb
```

2. Run the cells sequentially to:
   - Load PDF documents
   - Split text into chunks
   - Generate embeddings
   - Create vector store
   - Query with natural language

### Example Query

```python
question = "What is the transformer architecture?"
response = rag_chain.invoke(question)
print(response)
```

## Configuration

### LLM Options

The project supports multiple LLM providers:

**Groq (Fast, Free tier available)**
```python
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key="your_key")
```

**Ollama (Local, No API key)**
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

**OpenAI**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="your_key")
```

### Embedding Models

Default: `sentence-transformers/all-MiniLM-L6-v2`

Other options:
- `sentence-transformers/all-mpnet-base-v2` (better quality, slower)
- `BAAI/bge-small-en-v1.5` (good balance)

## Key Components

### 1. Document Loading
```python
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("path/to/document.pdf")
documents = loader.load()
```

### 2. Text Splitting
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)
```

### 3. Vector Store
```python
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
```

### 4. RAG Chain
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
```

## Notebooks

- **embeddings.ipynb**: Learn about embeddings, cosine similarity, and semantic search
- **chromadb.ipynb**: Complete RAG pipeline with all components
- **pdfdataparsing.ipynb**: PDF parsing techniques
- **Documents.ipynb**: Document loading examples

## Requirements

- Python 3.8+
- LangChain
- ChromaDB
- HuggingFace Transformers
- PyMuPDF
- Jupyter Notebook

See `requirements.txt` for full list.

## API Keys

Get free API keys:
- **Groq**: https://console.groq.com/keys (Fast inference, generous free tier)
- **HuggingFace**: https://huggingface.co/settings/tokens (Free)
- **OpenAI**: https://platform.openai.com/api-keys (Paid)


## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Embeddings from [HuggingFace](https://huggingface.co/)
- Vector store powered by [ChromaDB](https://www.trychroma.com/)
