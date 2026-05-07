import json
import os
import pandas as pd
import _compat  # noqa: F401 — Python 3.14 + urllib3 header encoding fix
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

EXCEL_PATH = "darija_nlp_database.xlsx"
OUTPUT_PATH = "darija_chunks.json"
HEADER_ROW_INDEX = 1  # Row 1 is the merged title; row 2 (index 1) is the real header

_UNICODE_MAP = str.maketrans({
    "—": "-", "–": "-",   # em / en dash
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "…": "...",                # ellipsis
})


def _sanitize(text: str) -> str:
    return text.translate(_UNICODE_MAP).encode("latin-1", errors="replace").decode("latin-1")


def _load_sheet(sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, header=HEADER_ROW_INDEX, dtype=str)
    df.fillna("", inplace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _build_intent_action_map(df_signals: pd.DataFrame) -> dict[str, str]:
    action_map: dict[str, str] = {}
    for _, row in df_signals.iterrows():
        intent = str(row.get("intent", "")).strip().lower()
        action = str(row.get("action", "")).strip()
        if intent:
            action_map[intent] = action
    return action_map


def build_raw_collection_chunks(df: pd.DataFrame, action_map: dict[str, str]) -> list[dict]:
    chunks = []
    df_valid = df[df["validated"].str.strip().str.lower() == "yes"]
    for _, row in df_valid.iterrows():
        intent = str(row.get("intent_category", "")).strip()
        text = _sanitize(
            f"[INTENT: {intent}]\n"
            f"User expression: {row.get('darija_arabizi', '')}\n"
            f"Variants: {row.get('variants', '')}\n"
            f"English: {row.get('english_meaning', '')}\n"
            f"French: {row.get('french_meaning', '')}\n"
            f"Action: {action_map.get(intent.lower(), '')}"
        )
        chunks.append({"text": text, "metadata": {
            "source": "darija_database",
            "intent": intent,
            "subject": str(row.get("subject", "")).strip(),
            "aggression": str(row.get("aggression_level", "")).strip(),
            "platform": str(row.get("source_platform", "")).strip(),
            "validated": True,
        }})
    return chunks


def build_conversation_chunks(df: pd.DataFrame) -> list[dict]:
    chunks = []
    df_valid = df[
        (df["use_in_rag"].str.strip().str.lower() == "yes")
        & (pd.to_numeric(df["quality_score"], errors="coerce").fillna(0) >= 4)
    ]
    for _, row in df_valid.iterrows():
        text = _sanitize(
            f"[VALIDATED DARIJA EXCHANGE]\n"
            f"Subject: {row.get('subject', '')}\n"
            f"Context: {row.get('context', '')}\n"
            f"User: {row.get('user_question', '')}\n"
            f"Assistant: {row.get('ideal_answer', '')}\n"
            f"Confirmation: {row.get('confirmation', '')}\n"
            f"Language mix: {row.get('language_mix', '')}"
        )
        try:
            quality_val = int(float(str(row.get("quality_score", "0")).strip()))
        except ValueError:
            quality_val = 0
        chunks.append({"text": text, "metadata": {
            "source": "conversation_thread",
            "subject": str(row.get("subject", "")).strip(),
            "quality": quality_val,
            "language_mix": str(row.get("language_mix", "")).strip(),
        }})
    return chunks


def sync_darija_db() -> dict:
    """
    Read the Excel file, build chunks from validated rows, save to JSON, and
    ingest into ChromaDB. Returns a summary dict with raw_count, conv_count, error.
    Can be called from the Streamlit app or from the command line via main().
    """
    df_raw     = _load_sheet("Raw Collection")
    df_threads = _load_sheet("Conversation Threads")
    df_signals = _load_sheet("Intent Signals Reference")

    action_map  = _build_intent_action_map(df_signals)
    raw_chunks  = build_raw_collection_chunks(df_raw, action_map)
    conv_chunks = build_conversation_chunks(df_threads)
    all_chunks  = raw_chunks + conv_chunks

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return {"raw_count": len(raw_chunks), "conv_count": len(conv_chunks),
                "error": "NVIDIA_API_KEY not set in .env"}

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=api_key,
    )
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    db.add_texts(
        texts=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )
    try:
        db.persist()
    except Exception:
        pass

    return {"raw_count": len(raw_chunks), "conv_count": len(conv_chunks), "error": None}


def main():
    result = sync_darija_db()
    if result["error"]:
        print(f"Error: {result['error']}")
        return
    print(
        f"Done -- ingested {result['raw_count']} Raw Collection chunks "
        f"and {result['conv_count']} Conversation Thread chunks into ChromaDB."
    )


if __name__ == "__main__":
    main()
