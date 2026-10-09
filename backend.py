from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
import tempfile
import rag_pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Autonomous MirAI Student Policy Advisor")

class ChatRequest(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to data directory with original filename to preserve metadata
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Ingest the document
        num_docs, num_chunks = rag_pipeline.ingest_document(file_path)
        
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "document_count": num_docs,
            "chunk_count": num_chunks
        }
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured in the environment.")
        
    # Verify handbook is indexed
    try:
        vs = rag_pipeline.get_vector_store()
        if len(vs.get()['ids']) == 0:
            raise HTTPException(status_code=400, detail="No documents ingested. Please ingest the policy handbook first.")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
         
    try:
        result = rag_pipeline.ask_question(request.question)
        return result
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Chat error: {error_msg}")
        if "quota" in error_msg.lower() or "authentication" in error_msg.lower():
             raise HTTPException(status_code=503, detail=f"Google API Error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {error_msg}")

@app.get("/health")
async def health_check():
    indexed = False
    try:
        if os.path.exists(rag_pipeline.CHROMA_DB_DIR):
            vs = rag_pipeline.get_vector_store()
            indexed = len(vs.get()['ids']) > 0
    except Exception as e:
        logger.warning(f"Health check vector store error: {str(e)}")
        
    return {
        "status": "healthy",
        "policy_handbook_indexed": indexed,
        "model": rag_pipeline.get_model_name()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
