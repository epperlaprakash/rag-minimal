# streamlit_app.py — Minimal Streamlit UI for the existing RAG

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# Reuse the working logic from the CLI script
from rag_cli import (
    ensure_dirs,
    load_index,
    build_index,
    read_txt_files,
    retrieve,
    build_prompt,
    llm_answer,
    DATA_DIR,
    TOP_K,
)

# ---------- Cached loader so the index builds/loads only once ----------
@st.cache_resource(show_spinner=True)
def get_rag_state():
    """Return (index, chunks, metas, embedder). Build if artifacts missing."""
    ensure_dirs()
    loaded = load_index()
    if loaded is not None:
        return loaded

    files = read_txt_files(DATA_DIR)
    if len(files) < 3:
        raise RuntimeError("Please add at least 3 .txt files to the data/ folder.")
    return build_index(files)


def one_line(s: str, max_len: int = 140) -> str:
    s = " ".join(s.split())
    return (s[:max_len] + "…") if len(s) > max_len else s


# ------------------------- UI -------------------------
st.set_page_config(page_title="Minimal RAG Demo", page_icon="🧠", layout="centered")

st.title("🧠 Minimal RAG Demo")
st.write("Ask questions *only* from the local `.txt` files in the `data/` folder. "
         "The model will say **“I don’t know”** if the answer isn’t in the documents.")

with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K chunks", min_value=1, max_value=5, value=TOP_K, help="How many chunks to retrieve")
    if st.button("🔁 Rebuild Index"):
        # Clear the cache so we rebuild embeddings and FAISS
        get_rag_state.clear()
        st.success("Cleared cached index. It will rebuild on next question.")

    st.markdown("**Quick tips**")
    st.markdown("- Make sure **Ollama** is running (or set `OPENAI_API_KEY`).")
    st.markdown("- Change any `.txt` in `data/` → click **Rebuild Index**.")

# Load or build the index
try:
    index, chunks, metas, embedder = get_rag_state()
except Exception as e:
    st.error(f"Setup error: {e}")
    st.stop()

# Sample prompts
with st.expander("💡 Try questions like"):
    st.markdown(
        "- What is the fastest land animal?\n"
        "- Which planet has the Great Red Spot?\n"
        "- What did Gutenberg invent?"
    )

# Question input
q = st.text_input("Your question", placeholder="Type a question about your documents…")
ask = st.button("Ask")

if ask and q.strip():
    with st.spinner("Thinking…"):
        hits = retrieve(q.strip(), index, embedder, chunks, metas, k=k)
        prompt = build_prompt(q.strip(), hits)
        answer = llm_answer(prompt)

    st.subheader("Answer")
    st.markdown(f"> {answer}")

    # Nicely formatted sources (dedupe by file, keep best rank)
    st.subheader("Sources")
    seen = {}
    for r in hits:
        file = r["meta"]["file"]
        if file not in seen:
            seen[file] = r

    if not seen:
        st.info("No sources (retrieval returned nothing).")
    else:
        for rank, (file, r) in enumerate(seen.items(), start=1):
            score_pct = f"{r['score']*100:.1f}%"
            snippet = one_line(r["chunk"])
            st.markdown(f"**{rank}. [{file}]** — {score_pct}")
            st.caption(snippet)
else:
    st.info("Enter a question and click **Ask**.")
