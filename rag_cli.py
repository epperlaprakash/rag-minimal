"""
rag_cli.py — a tiny Retrieval-Augmented Generation (RAG) demo
Run:
  1) Activate venv
  2) python rag_cli.py
Then type your questions. Type 'exit' to quit.
"""

import os
import glob
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Embeddings
from sentence_transformers import SentenceTransformer
# Vector store
import faiss

# ---------- Settings ----------
DATA_DIR = "data"           # where .txt files live
ART_DIR = "artifacts"       # where we save the index and metadata
CHUNK_SIZE = 800            # characters per chunk
CHUNK_OVERLAP = 120         # how much to overlap chunks
TOP_K = 4                   # how many chunks to retrieve each question
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = os.path.join(ART_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(ART_DIR, "chunks.pkl")
METAS_PATH = os.path.join(ART_DIR, "metas.pkl")
TOP_K = 3


# ============== Small helpers ==============
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ART_DIR,  exist_ok=True)

def _one_line(s: str, max_len: int = 140) -> str:
    """Collapse whitespace/newlines and trim for display."""
    s = " ".join(s.split())
    return (s[:max_len] + "…") if len(s) > max_len else s


# ============== Data loading & chunking ==============
def read_txt_files(folder: str) -> List[Tuple[str, str]]:
    """Return list of (filename, full_text) for all .txt files."""
    out = []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            out.append((os.path.basename(path), f.read()))
    return out

def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[str, int, int]]:
    """Return list of (chunk_text, start_idx, end_idx) with overlap."""
    chunks: List[Tuple[str, int, int]] = []
    if size <= 0:
        return [(text, 0, len(text))]
    step = max(1, size - max(0, overlap))
    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        ch = text[start:end]
        if ch:
            chunks.append((ch, start, end))
        if end >= len(text):
            break
    return chunks


# ============== Index build/load ==============
def build_index(files: List[Tuple[str, str]]):
    """Embed all chunks and build a FAISS index. Save artifacts."""
    print("• Loading embedding model…")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("• Chunking text files…")
    all_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []  # {'file': str, 'start': int, 'end': int}
    for fname, txt in files:
        for ch, start, end in chunk_text(txt, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append(ch)
            metas.append({"file": fname, "start": start, "end": end})

    print(f"  - Total chunks: {len(all_chunks)}")
    if not all_chunks:
        raise SystemExit("No text found. Add .txt files to the data/ folder.")

    print("• Creating embeddings (this may take a moment)…")
    embs = embedder.encode(
        all_chunks, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    print("• Building FAISS index…")
    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized vectors
    index.add(embs)

    print("• Saving artifacts…")
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(METAS_PATH, "wb") as f:
        pickle.dump(metas, f)

    print("✓ Index ready.")
    return index, all_chunks, metas, embedder

def load_index():
    """Load FAISS index + metadata if present, else return None."""
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(METAS_PATH)):
        return None
    print("• Loading saved artifacts…")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(METAS_PATH, "rb") as f:
        metas = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print("✓ Loaded existing index.")
    return index, chunks, metas, embedder


# ============== Retrieval ==============
def retrieve(query: str, index, embedder, chunks: List[str], metas: List[Dict[str, Any]], k: int = TOP_K):
    """Return list of {'score': float, 'meta': {...}, 'chunk': str} for top-k."""
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, k)
    results = []
    for i, s in zip(idxs[0], scores[0]):
        results.append({"score": float(s), "meta": metas[i], "chunk": chunks[i]})
    return results


# ============== Prompting & LLMs ==============
def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([f"[{r['meta']['file']}] {r['chunk']}" for r in retrieved])
    return f"""You are a careful assistant.
Use ONLY the context below. If the context does not contain the answer, say exactly: "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer briefly (2–5 sentences) and include file-name citations in square brackets, e.g., [animals.txt].
"""

def answer_with_openai(prompt: str) -> str:
    """Use OpenAI if OPENAI_API_KEY is set."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "(No OPENAI_API_KEY set; skipping OpenAI.)"
    try:
        import openai  # optional dependency
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using only the provided context. Be concise and cite sources."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

def answer_with_ollama(prompt: str) -> str:
    """Call a local Ollama model (llama3 by default)."""
    try:
        import subprocess
        run = subprocess.run(
            ["ollama", "run", "llama3"],   # change to "llama3.2" if you prefer smaller/faster
            input=prompt.encode("utf-8"),  # send prompt via stdin (compatible)
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
        if run.returncode != 0:
            return f"(Ollama error: {run.stderr.decode('utf-8', errors='ignore')})"
        return run.stdout.decode("utf-8", errors="ignore").strip()
    except subprocess.TimeoutExpired:
        return "(Ollama timed out. Try again or switch to a smaller model like llama3.2.)"
    except FileNotFoundError:
        return "(Ollama not installed.)"
    except Exception as e:
        return f"(Ollama error: {e})"

def llm_answer(prompt: str) -> str:
    """Prefer OpenAI if key is set, else use Ollama."""
    if os.getenv("OPENAI_API_KEY"):
        return answer_with_openai(prompt)
    return answer_with_ollama(prompt)


# ============== CLI ==============
def main():
    print("== Minimal RAG Demo ==")
    ensure_dirs()

    loaded = load_index()
    if loaded is None:
        print("• No saved index found. Building from data/…")
        files = read_txt_files(DATA_DIR)
        if len(files) < 3:
            print("Please add at least 3 .txt files into the 'data' folder and run again.")
            return
        index, chunks, metas, embedder = build_index(files)
    else:
        index, chunks, metas, embedder = loaded

    print("\nRAG is ready. Type your question (or 'exit').")
    print("Try questions like:")
    print("  • What is the fastest land animal?")
    print("  • Which planet has the Great Red Spot?")
    print("  • What did Gutenberg invent?")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not q:
            continue

        hits = retrieve(q, index, embedder, chunks, metas, k=TOP_K)
        prompt = build_prompt(q, hits)
        answer = llm_answer(prompt)

        print("\n--- Answer ---")
        print(answer)

        # Nicely formatted sources (dedupe by file, show best score)
        print("\nSources:")
        seen: Dict[str, Dict[str, Any]] = {}
        for r in hits:
            file = r["meta"]["file"]
            if file not in seen:
                seen[file] = r  # keep the first/highest-ranked hit per file

        for rank, (file, r) in enumerate(seen.items(), start=1):
            score_pct = f"{r['score']*100:.1f}%"
            snippet = _one_line(r["chunk"], max_len=140)
            print(f"  {rank}. [{file}]  ({score_pct})  {snippet}")
        print()

if __name__ == "__main__":
    main()