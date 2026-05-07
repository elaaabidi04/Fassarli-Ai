import json
import os
import _compat  # noqa: F401 — Python 3.14 + urllib3 header encoding fix
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

CHUNKS_PATH = "darija_chunks.json"
CHROMA_PATH = "./chroma_db"


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    raw_count  = sum(1 for c in chunks if c["metadata"].get("source") == "darija_database")
    conv_count = sum(1 for c in chunks if c["metadata"].get("source") == "conversation_thread")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY not set — check your .env file.")

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=api_key,
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    db.add_texts(texts=texts, metadatas=metadatas)

    try:
        db.persist()
    except Exception:
        pass

    print(
        f"Ingested {raw_count} chunks from Raw Collection, "
        f"{conv_count} chunks from Conversation Threads"
    )


if __name__ == "__main__":
    main()
