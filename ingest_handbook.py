import os
import argparse
from rag_pipeline import ingest_document, CHROMA_DB_DIR

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF handbook into the RAG vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found at {args.pdf_path}")
        return
        
    print(f"Ingesting {args.pdf_path} into {CHROMA_DB_DIR}...")
    try:
        num_docs, num_chunks = ingest_document(args.pdf_path)
        print(f"Successfully ingested {num_docs} document(s) resulting in {num_chunks} chunk(s).")
    except Exception as e:
        print(f"Failed to ingest document: {str(e)}")

if __name__ == "__main__":
    main()
