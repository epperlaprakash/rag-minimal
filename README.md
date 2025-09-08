# Minimal RAG Prototype

A tiny Retrieval-Augmented Generation (RAG) demo in Python.  
It loads a few `.txt` files, splits them into chunks, builds a vector index (FAISS), retrieves the most relevant chunks, and asks an LLM (Ollama by default) to answer using only that context.

---

## ✨ Features
- Loads & embeds 3+ text files from `data/`
- Overlapping chunking
- Vector search with FAISS
- Simple CLI (type a question → get an answer)
- Web UI with Streamlit (alternative to CLI)
- Shows sources with score % + snippet
- “I don’t know” guardrail when context is missing
- LLM: Ollama (Llama 3) by default; optional OpenAI API

---

## 📂 Project Structure
```
rag-minimal/
├─ data/            # .txt knowledge files
│  ├─ animals.txt
│  ├─ planets.txt
│  └─ history.txt
├─ artifacts/       # auto-saved index & metadata (created on first run)
├─ rag_cli.py       # main script (CLI version)
├─ streamlit_app.py # Streamlit web UI version
├─ README.md
├─ requirements.txt
└─ .gitignore
```

---

## ⚙️ Requirements
- Python 3.10+
- Python packages:
```bash
pip install -r requirements.txt
```
- One LLM path:
  - Ollama (local, free) — recommended, or
  - OpenAI API key (no local install)

`requirements.txt`
```text
sentence-transformers
faiss-cpu
numpy
streamlit
# openai  # optional; only if you want to use OpenAI
```

---

## 🧠 Choose your LLM

### Option A — Ollama (local, free)
Download & install: https://ollama.com/download  
Open the Ollama app once so the service starts.  
In Terminal, pull a model:
```bash
ollama pull llama3.2
```
Quick test:
```bash
ollama run llama3.2 "hello"
```

### Option B — OpenAI (no local install)
Create an API key: https://platform.openai.com → View API keys → Create new key  

Set the key in your shell:

**macOS/Linux**
```bash
export OPENAI_API_KEY="sk-..."
```

**Windows PowerShell**
```bash
$env:OPENAI_API_KEY="sk-..."
```

The script will auto-use OpenAI if `OPENAI_API_KEY` is set; otherwise it uses Ollama.

---

## ▶️ Run (two options)

### Option 1 — CLI (terminal)
```bash
python3 rag_cli.py   # macOS/Linux
python rag_cli.py    # Windows
```

### Option 2 — Streamlit (web UI)
```bash
streamlit run streamlit_app.py
```
The UI opens at http://localhost:8501.  
Use the **Top-K Chunks** slider and **Rebuild Index** (sidebar) if you change files in `data/`.

---

## 💻 Example (CLI or Streamlit)

```text
== Minimal RAG Demo ==
RAG is ready. Type your question (or 'exit').
Try questions like:
  • What is the fastest land animal?
  • Which planet has the Great Red Spot?
  • What did Gutenberg invent?
```

**Q:** What is the fastest land animal?  
**Answer:** According to the provided documents, the fastest land animal is the cheetah... [animals.txt].  

Sources:
1. [animals.txt] (67.4%) Cheetahs are the fastest land animals, capable of short bursts up to about 100–120 km/h…  
2. [planets.txt] (18.9%) Olympus Mons, is on Mars. Jupiter is the largest planet, a gas giant with strong bands…  

---

## 🙅 “I don’t know” behavior

**Q:** Who is the president of the United States?  
**Answer:** I don't know based on the provided documents. The context only contains information about planets and animals…  

Sources:
1. [planets.txt] (15.7%) Olympus Mons, is on Mars…  
2. [animals.txt] (6.4%) Cheetahs are the fastest land animals…  

---

## 🔄 Rebuilding the index
If you change the `.txt` files in the `data/` folder, delete the files in `artifacts/` and rerun either CLI or Streamlit.


---

## 🧪 Tools / Models Used
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS (inner product on normalized vectors = cosine)
- LLM: Ollama Llama 3.2 by default (or OpenAI if `OPENAI_API_KEY` is set)
- Language: Python 3.11
- Web UI: Streamlit

---

## ⏱️ Time Spent
~4–5 hours (setup, learning, coding, testing).
