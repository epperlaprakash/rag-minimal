Minimal RAG Prototype

A tiny Retrieval-Augmented Generation (RAG) demo in Python.
It loads a few .txt files, splits them into chunks, builds a vector index (FAISS), retrieves the most relevant chunks, and asks an LLM (Ollama by default) to answer using only that context.

**Features:**
    Loads & embeds 3+ text files from data/
    Overlapping chunking
    Vector search with FAISS
    Simple CLI (type a question → get an answer)
    Shows sources with score % + snippet
    “I don’t know” guardrail when context is missing
    LLM: Ollama (Llama 3) by default; optional OpenAI API

📂 **Project Structure**
rag-minimal/
├─ data/            # .txt knowledge files
│  ├─ animals.txt
│  ├─ planets.txt
│  └─ history.txt
├─ artifacts/       # auto-saved index & metadata (created on first run)
├─ rag_cli.py       # main script
├─ README.md
├─ requirements.txt
└─ .gitignore


**Requirements:**
    Python 3.10+
    Python packages: 

    ```bash 
    pip install -r requirements.txt
    ```

    One LLM path:
        Ollama (local, free) — recommended, or
        OpenAI API key (no local install)

requirements.txt

sentence-transformers
faiss-cpu
numpy
# openai  # optional; only if you want to use OpenAI

🧠 **Choose your LLM**
Option A — Ollama (local, free)
Download & install: https://ollama.com/download
Open the Ollama app once so the service starts.
In Terminal, pull a model:

```bash
ollama pull llama3
```

Quick test:
```bash
ollama run llama3 "hello"
```


# Option B — OpenAI (no local install)

Create an API key: https://platform.openai.com
 → View API keys → Create new key

Set the key in your shell:

    macOS/Linux:

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```


    Windows PowerShell:

    ```bash
    $env:OPENAI_API_KEY="sk-..."
    ```


The script will auto-use OpenAI if OPENAI_API_KEY is set; otherwise it uses Ollama.

⚙️ **Setup**
macOS / Linux

```bash 
cd rag-minimal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell)

```bash
cd rag-minimal
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# If activation is blocked:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Put at least 3 .txt files in data/ (the repo includes examples).

▶️ Run
```bash
python3 rag_cli.py 
```    # macOS/Linux
# or
```bash
python rag_cli.py      # Windows
```

You’ll see:

== Minimal RAG Demo ==
RAG is ready. Type your question (or 'exit').
Try questions like:
  • What is the fastest land animal?
  • Which planet has the Great Red Spot?
  • What did Gutenberg invent?
> 

Example
> What is the fastest land animal?

--- Answer ---
According to the provided documents, the fastest land animal is the cheetah... [animals.txt].

Sources:
  1. [animals.txt]  (67.4%)  Cheetahs are the fastest land animals, capable of short bursts up to about 100–120 km/h…
  2. [planets.txt]  (18.9%)  Olympus Mons, is on Mars. Jupiter is the largest planet, a gas giant with strong bands…

“I don’t know” behavior
> Who is the president of the United States?

--- Answer ---
I don't know based on the provided documents. The context only contains information about planets and animals…

Sources:
  1. [planets.txt]  (15.7%)  Olympus Mons, is on Mars…
  2. [animals.txt]  (6.4%)   Cheetahs are the fastest land animals…

# Rebuild the index if and when you change the .txt files in data folder. simply delete the files in artifacts folder and then rerun the rag_cli.py


**🧪 Tools / Models Used**
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector store: FAISS (inner product on normalized vectors = cosine)
LLM: Ollama Llama 3 by default (or OpenAI if OPENAI_API_KEY is set)
Language: Python 3.11

⏱️ **Time Spent**
~4–5 hours (setup, learning, coding, testing).