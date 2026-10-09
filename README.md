# Autonomous MirAI Student Policy Advisor

A production-grade Retrieval-Augmented Generation (RAG) system built to act as an autonomous policy advisor for MirAI School of Technology students.

## Architecture

This project implements a complete RAG pipeline:
1. **Document Ingestion**: `PyPDFLoader` extracts text from the PDF handbook.
2. **Semantic Chunking**: `RecursiveCharacterTextSplitter` chunks the text while preserving context.
3. **Embeddings & Storage**: `GoogleGenerativeAIEmbeddings` (text-embedding-004) vectorizes the chunks and stores them in a local ChromaDB instance.
4. **Advanced Retrieval**: A `MultiQueryRetriever` expands and rewrites user queries for better semantic matching against formal handbook terminology.
5. **Generation**: `ChatGoogleGenerativeAI` (gemini-3.8-flash, configurable) generates answers with strict guardrails to prevent hallucination.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Google API key for Gemini

### 2. Installation
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and add your `GOOGLE_API_KEY`.

*(Note: `LLM_MODEL` defaults to `gemini-1.5-flash` in the provided `.env.example` in case `gemini-3.8-flash` is not available yet in your region/version.)*

### 4. Data Ingestion
Place the PDF handbook (`Mirai_SoT_Policy_Handbook_2026.pdf`) in the `data/` directory, then ingest it:
```bash
python3 ingest_handbook.py data/Mirai_SoT_Policy_Handbook_2026.pdf
```
Alternatively, you can ingest the document through the Streamlit UI once the backend is running.

### 5. Running the Servers

**Start the FastAPI Backend:**
```bash
uvicorn backend:app --reload --port 8000
```
*The backend provides `/ingest`, `/chat`, and `/health` endpoints.*

**Start the Streamlit Frontend (in a new terminal):**
```bash
source venv/bin/activate
streamlit run frontend.py
```
*The Streamlit UI will be available at `http://localhost:8501`.*

### 6. Running Tests & Evaluation
To run the manual certification audit tests:
```bash
python3 test_rag.py
```

To run the automated LLM-as-a-judge evaluation:
```bash
python3 evaluate.py
```
*Scores will be saved to `rag_eval_scores.csv`.*
