from fastapi import FastAPI, Query
from pydantic import BaseModel
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import datetime

INDEX_DIR = Path("artifacts/faiss_index")
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    
    model_config = {
        'protected_namespaces': ()
    }

class QueryResponse(BaseModel):
    query: str
    results: List[Dict]
    context: str
    timestamp: str
    metadata: Dict

app = FastAPI(title="PDF RAG Service", version="1.0.0")

def load_index():
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    mapping = json.loads((INDEX_DIR / "mapping.json").read_text())
    texts = [f"Doc {m['doc_id']} — chunk {m['chunk_id']} (load your real text here)" for m in mapping]
    return index, mapping, texts

INDEX, MAPPING, TEXTS = load_index()
EMB = SentenceTransformer(EMBEDDING_MODEL)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=QueryResponse)
def ask(body: QueryRequest):
    vec = EMB.encode([body.query], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
    sims, idxs = INDEX.search(vec, body.top_k)
    idx_list = idxs[0].tolist()
    
    # Get full text content from mapping
    results = []
    context_texts = []
    
    for i, idx in enumerate(idx_list):
        doc_info = MAPPING[idx]
        text = doc_info.get("text", "[Text not found]")
        context_texts.append(text)
        results.append({
            "rank": i+1,
            "doc_id": doc_info["doc_id"],
            "chunk_id": doc_info["chunk_id"],
            "score": float(sims[0][i]),
            "text": text,
            "preview": text[:200] + "..." if len(text) > 200 else text
        })
    
    # Combine context for better readability
    combined_context = "\n\n---\n\n".join(context_texts)
    
    return QueryResponse(
        query=body.query,
        results=results,
        context=combined_context,
        timestamp=datetime.now().isoformat(),
        metadata={
            "num_results": len(results),
            "model": EMBEDDING_MODEL,
            "top_k": body.top_k
        }
    )
