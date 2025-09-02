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
    page_num: int  # Add page number tracking

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

def read_text_from_pdf(path: Path) -> List[Dict[str, str]]:
    """Extract text from PDF with page numbers."""
    try:
        doc = fitz.open(path)
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                pages.append({
                    "text": text,
                    "page": page_num + 1
                })
        return pages
    except Exception as e:
        logger.error(f"Failed to extract text from {path}: {e}")
        return [{"text": f"[ERROR extracting {path.name}]", "page": 1}]

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
def task_load_pdfs(data_dir: str) -> Dict[str, List[Dict[str, str]]]:
    paths = sorted(Path(data_dir).glob("*.pdf"))
    docs = {}
    for p in paths:
        docs[p.stem] = read_text_from_pdf(p)
    return docs

@task
def task_chunk_docs(docs: Dict[str, List[Dict[str, str]]], chunk_size: int, overlap: int) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    chunk_id = 0
    
    for doc_id, pages in docs.items():
        for page in pages:
            text = page["text"]
            page_num = page["page"]
            
            # Split page text into chunks
            sentences = text.split(". ")
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip() + ". "
                words = sentence.split()
                
                if current_length + len(words) <= chunk_size:
                    current_chunk.append(sentence)
                    current_length += len(words)
                else:
                    if current_chunk:  # Save current chunk
                        chunks.append(DocChunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text="".join(current_chunk),
                            page_num=page_num
                        ))
                        chunk_id += 1
                        
                        # Handle overlap
                        if overlap > 0:
                            # Keep last few sentences for overlap
                            overlap_text = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                            current_chunk = overlap_text
                            current_length = sum(len(t.split()) for t in current_chunk)
                        else:
                            current_chunk = []
                            current_length = 0
                    
                    current_chunk.append(sentence)
                    current_length = len(words)
            
            # Add remaining text as final chunk
            if current_chunk:
                chunks.append(DocChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text="".join(current_chunk),
                    page_num=page_num
                ))
                chunk_id += 1
    
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
    ensure_dir(index_dir)
    
    # Save FAISS index
    index_path = Path(index_dir) / "index.faiss"
    mapping_path = Path(index_dir) / "mapping.json"
    
    faiss.write_index(index, str(index_path))
    
    # Save mapping with page numbers
    mapping = [{
        "doc_id": c.doc_id,
        "chunk_id": c.chunk_id,
        "text": c.text,
        "page": c.page_num
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
def summarize_chunk(text: str, max_words: int = 50) -> str:
    """Create a brief summary of chunk text, preserving sentence boundaries."""
    sentences = text.split(". ")
    words = []
    total_words = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        if total_words + len(sentence_words) <= max_words:
            words.extend(sentence_words)
            total_words += len(sentence_words)
        else:
            break
    
    summary = " ".join(words)
    return summary + "..." if total_words == max_words else summary

def generate_final_answer(query: str, results: List[Dict]) -> str:
    """Generate a structured final answer from search results."""
    answer_parts = []
    answer_parts.append(f"Based on {len(results)} relevant sections, here's a travel plan:")
    
    # Extract key information
    answer_parts.append("\n1. Recommended Activities:")
    for r in results[:2]:  # Use top 2 results for activities
        if "Things to Do" in r["doc_id"]:
            summary = summarize_chunk(r["text"], 75)
            answer_parts.append(f"   • {summary}")
    
    # Add dining recommendations
    answer_parts.append("\n2. Dining Options:")
    for r in results:
        if "Cuisine" in r["doc_id"] or "Restaurants" in r["doc_id"]:
            summary = summarize_chunk(r["text"], 75)
            answer_parts.append(f"   • {summary}")
    
    # Add tips
    answer_parts.append("\n3. Travel Tips:")
    for r in results:
        if "Tips" in r["doc_id"]:
            summary = summarize_chunk(r["text"], 75)
            answer_parts.append(f"   • {summary}")
    
    return "\n".join(answer_parts)

def query(text: str, top_k: int = 5) -> Dict:
    """Enhanced query with summaries and page info."""
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
        
        # Enhanced results with summaries
        results = []
        context_texts = []
        for i, idx in enumerate(idxs[0]):
            doc_info = mapping[idx]
            full_text = doc_info.get("text", "[Text not found]")
            page_num = doc_info.get("page", 1)  # Default to page 1 if not found
            
            results.append({
                "rank": i+1,
                "doc_id": doc_info["doc_id"],
                "page": page_num,
                "score": float(sims[0][i]),
                "summary": summarize_chunk(full_text)
            })
            context_texts.append(full_text)
        
        # Generate final answer with increased word limit
        final_answer = f"Based on {len(results)} relevant sections from the documents:\n"
        final_answer += "\nRecommended answer (250 words max): "
        final_answer += summarize_chunk(" ".join(context_texts), max_words=250)
        
        return {
            "results": results,
            "answer": final_answer
        }
        
    except Exception as e:
        logging.error(f"Error during query: {str(e)}")
        raise RuntimeError(
            f"Failed to query index: {str(e)}. "
            "Try running the pipeline again: python run_pipeline.py"
        )
