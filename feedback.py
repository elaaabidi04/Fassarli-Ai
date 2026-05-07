from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from datetime import datetime

def save_good_exchange(question: str, answer: str) -> str:
    """
    Persist a validated Q&A pair as a new document in ChromaDB.
    Returns the document ID so it can be deleted later if needed.
    """
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    doc_text = (
        f"[VALIDATED DARIJA EXCHANGE — {datetime.now().date()}]\n"
        f"User: {question}\n"
        f"Assistant: {answer}"
    )

    ids = db.add_texts(
        texts=[doc_text],
        metadatas=[{
            "source": "feedback_loop",
            "type": "validated_exchange",
            "date": str(datetime.now().date()),
        }],
    )

    try:
        db.persist()  # no-op on newer Chroma versions (auto-persisted)
    except Exception:
        pass

    return ids[0]


def delete_exchange(doc_id: str):
    """Remove a previously saved exchange from ChromaDB by its document ID."""
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    db.delete(ids=[doc_id])
    try:
        db.persist()
    except Exception:
        pass
