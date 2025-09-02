# PDF RAG — MLOps Refactor

## Quick Setup Guide

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
```bash
mkdir -p data/pdfs
# Copy your PDF files into data/pdfs/
```

3. Run the pipeline (choose one method):
   - Method A: Run Python script
     ```bash
     python run_pipeline.py
     ```
   - Method B: Use Jupyter notebook
     - Open `Adobe_PDF_Outline_Extractor_MLOps.ipynb`
     - Go to Section 7 and run `rag_flow()`

The pipeline will:
- Process PDFs from data/pdfs/
- Create embeddings and FAISS index
- Save artifacts to artifacts/
- Log metrics to MLflow

## API Usage

Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Test endpoints:
- Health check: GET http://localhost:8080/health
- Search: POST http://localhost:8080/ask
  ```json
  {
    "query": "your search query",
    "top_k": 5
  }
  ```

## Docker Deployment

Build and run container:
```bash
# Build image
docker build -t pdf-rag:latest .

# Run container
docker run -p 8080:8080 pdf-rag:latest
```
