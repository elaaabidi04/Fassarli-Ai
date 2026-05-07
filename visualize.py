import streamlit as st
import chromadb
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import umap
import os

load_dotenv()

st.set_page_config(page_title="Vector DB Explorer", layout="wide")
st.title("Vector DB Explorer")

# ── Pipeline diagram ──────────────────────────────────────────────────────────
st.markdown("## How the pipeline works")
st.markdown("""
```
PDF file
   │
   ▼
PyPDFLoader  ──►  raw pages
   │
   ▼
RecursiveCharacterTextSplitter  ──►  chunks  (500 chars, 50 overlap)
   │
   ▼
NVIDIAEmbeddings  ──►  vectors  (high-dimensional floats)
   │
   ▼
ChromaDB  (stored on disk at ./chroma_db)
   │
   ▼  (at query time)
Similarity search  ──►  top-k relevant chunks
   │
   ▼
ChatNVIDIA (Llama 3.3)  ──►  answer
```
""")

st.divider()

# ── Load data directly via chromadb client (bypasses LangChain wrapper bugs) ──
st.markdown("## Stored chunks & embeddings")

DB_PATH = "./chroma_db"

if not os.path.exists(DB_PATH):
    st.warning("No `chroma_db` folder found. Upload and index a PDF in the main app first.")
    st.stop()

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    collections = client.list_collections()

    if not collections:
        st.warning("ChromaDB exists but has no collections yet. Index a PDF in the main app first.")
        st.stop()

    collection_name = collections[0].name
    collection = client.get_collection(collection_name)
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    docs = data["documents"]
    metas = data["metadatas"]
    vecs = data["embeddings"]

    if not docs:
        st.warning("Collection is empty. Upload and index a PDF in the main app first.")
        st.stop()

    st.success(f"{len(docs)} chunks loaded from collection **{collection_name}**")

    # ── Chunk table ───────────────────────────────────────────────────────────
    st.markdown("### Chunks")
    df = pd.DataFrame({
        "Chunk #": range(1, len(docs) + 1),
        "Source": [m.get("source", "?") for m in metas],
        "Page": [m.get("page", "?") for m in metas],
        "Length (chars)": [len(d) for d in docs],
        "Text preview": [d[:120] + "..." if len(d) > 120 else d for d in docs],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── 2-D embedding plot ────────────────────────────────────────────────────
    st.markdown("### Embedding space (2-D UMAP projection)")
    st.caption("Each dot is one chunk. Hover to read it. Nearby dots have similar meaning.")

    if len(docs) < 2:
        st.info("Need at least 2 chunks to plot. Index more content first.")
        st.stop()

    with st.spinner("Running UMAP dimensionality reduction..."):
        n_neighbors = min(15, len(docs) - 1)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        coords = reducer.fit_transform(vecs)

    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Chunk #": range(1, len(docs) + 1),
        "Source": [m.get("source", "?") for m in metas],
        "Page": [str(m.get("page", "?")) for m in metas],
        "Text": [d[:200] + "..." if len(d) > 200 else d for d in docs],
    })

    fig = px.scatter(
        plot_df,
        x="x", y="y",
        color="Source",
        hover_data={"Chunk #": True, "Page": True, "Text": True, "x": False, "y": False},
        labels={"color": "Source"},
        height=600,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not load ChromaDB: {e}")
    st.info("Make sure you have indexed a PDF in the main app first.")
