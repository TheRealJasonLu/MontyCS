# MontyCS RAG Assistant

Local AI assistant for Montgomery Computer Science Club.

## Setup

1. Install dependencies:
pip install -r requirements.txt

2. Install Ollama:
https://ollama.com

3. Pull models:
ollama pull mistral
ollama pull nomic-embed-text

4. Build database:
python ingest.py

5. Run app:
streamlit run app.py