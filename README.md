# MontyCS Assistant

A fully local Retrieval-Augmented Generation (RAG) assistant for the Montgomery Computer Science Club.

This project allows users to ask questions about club information (meetings, events, resources, etc.) and receive AI-generated answers grounded in local documents.

---

## Features

- Runs **100% locally** (no API keys required)
- Uses **Ollama** for local LLM + embeddings
- Uses **Chroma** as a vector database
- Simple **Streamlit UI**
- Supports custom club documents

---

## How It Works

1. Documents are stored in `data/docs/`
2. `ingest.py`:
   - Loads documents
   - Splits into chunks
   - Creates embeddings
   - Stores them in a local vector database (`vectordb/`)
3. `app.py`:
   - Takes a user query
   - Retrieves relevant document chunks
   - Sends them to a local LLM
   - Returns an answer with sources

---

## Setup Instructions

### 1. Clone the repo
git clone https://github.com/TheRealJasonLu/MontyCS.git
cd MontyCS

### 2. Create virtual environment

python -m venv .venv
..venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Install Ollama

Download and install:
https://ollama.com

### 5. Download models

ollama pull mistral
ollama pull nomic-embed-text

### 6. Build vector database

python ingest.py

### 7. Run the app

streamlit run app.py

---

## Author

Jason Lu  
Montgomery Computer Science Club