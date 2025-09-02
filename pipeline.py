# Core imports
import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import logging
import warnings

# Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# ML imports
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import mlflow
from prefect import task, flow

# Constants
PROJECT_NAME = "pdf-rag"
EXPERIMENT_NAME = "pdf-rag-experiments"
RUN_NAME = "mpnet-faiss-baseline"
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5

# Paths
DATA_DIR = "data/pdfs"
ARTIFACTS_DIR = "artifacts"
INDEX_DIR = f"{ARTIFACTS_DIR}/faiss_index"

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# Add PDF extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF for PDF extraction...")
    import subprocess
    subprocess.check_call(["pip", "install", "PyMuPDF"])
    import fitz

logger = logging.getLogger(__name__)

def read_text_from_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {path}: {e}")
        return f"[ERROR extracting {path.name}]"

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, trying to break at sentence boundaries."""
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into sentences (rough approximation)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            if chunks and overlap > 0:
                # Take last few sentences that fit within overlap
                overlap_text = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_len = len(sent.split())
                    if overlap_length + sent_len <= overlap:
                        overlap_text.insert(0, sent)
                        overlap_length += sent_len
                    else:
                        break
                current_chunk = overlap_text
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
            
            # Add current sentence to the new chunk
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

@task
def task_load_pdfs(data_dir: str) -> Dict[str, str]:
    paths = sorted(Path(data_dir).glob("*.pdf"))
    if not paths:
        logger.warning(f"No PDFs found in {data_dir}")
        raise RuntimeError(f"No PDFs found in {data_dir}. Please add PDFs first.")
    
    docs = {}
    for p in paths:
        docs[p.stem] = read_text_from_pdf(p)
    logger.info(f"Loaded {len(docs)} documents")
    return docs

@task
def task_chunk_docs(docs: Dict[str, str], chunk_size: int, overlap: int) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for doc_id, text in docs.items():
        parts = text.split()  # Simple split for demo
        chunk = DocChunk(doc_id=doc_id, chunk_id=0, text=text)
        chunks.append(chunk)
    return chunks

@task
def task_embed_chunks(chunks: List[DocChunk], model_name: str) -> Tuple[np.ndarray, List[DocChunk]]:
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype('float32'), chunks

@task
def task_build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index from embeddings."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

@task
def task_save_artifacts(index: faiss.Index, chunks: List[DocChunk], index_dir: str) -> str:
    """Save FAISS index and mapping with text content."""
    ensure_dir(index_dir)
    
    # Save FAISS index
    index_path = Path(index_dir) / "index.faiss"
    mapping_path = Path(index_dir) / "mapping.json"
    
    # Save index directly (no dict wrapping needed)
    faiss.write_index(index, str(index_path))
    
    # Save mapping with text content
    mapping = [{
        "doc_id": c.doc_id,
        "chunk_id": c.chunk_id,
        "text": c.text
    } for c in chunks]
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    return str(Path(index_dir).resolve())

@flow(name=PROJECT_NAME)
def rag_flow(
    data_dir: str = DATA_DIR,
    model_name: str = EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    top_k: int = TOP_K
):
    ensure_dir(ARTIFACTS_DIR)
    
    # Pipeline steps
    docs = task_load_pdfs(data_dir)
    chunks = task_chunk_docs(docs, chunk_size, chunk_overlap)
    embeddings, chunks = task_embed_chunks(chunks, model_name)
    index = task_build_index(embeddings)  # Returns faiss.Index directly
    index_path = task_save_artifacts(index, chunks, INDEX_DIR)
    
    metrics = {
        "num_docs": len(docs),
        "num_chunks": len(chunks)
    }
    return metrics

# Query interface
def query(text: str, top_k: int = 5) -> List[Dict]:
    """Query the RAG system with proper error handling."""
    index_path = INDEX_DIR
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    if not Path(index_path).exists():
        raise RuntimeError(
            f"Index directory not found at {index_path}. "
            "Please run the pipeline first: python run_pipeline.py"
        )
    
    try:
        # Load index and mapping
        index = faiss.read_index(str(Path(index_path) / "index.faiss"))
        with open(Path(index_path) / "mapping.json", encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Encode query
        q_emb = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype('float32')
        
        # Search
        sims, idxs = index.search(q_emb, top_k)
        
        # Format results with safe text access
        results = []
        for i, idx in enumerate(idxs[0]):
            doc_info = mapping[idx]
            results.append({
                "rank": i+1,
                "doc_id": doc_info["doc_id"],
                "chunk_id": doc_info["chunk_id"],
                "text": doc_info.get("text", "[Text not found]"),
                "score": float(sims[0][i])
            })
        return results
        
    except Exception as e:
        logging.error(f"Error during query: {str(e)}")
        raise RuntimeError(
            f"Failed to query index: {str(e)}. "
            "Try running the pipeline again: python run_pipeline.py"
        )
