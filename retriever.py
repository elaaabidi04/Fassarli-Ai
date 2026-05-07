from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_INTENT_INSTRUCTIONS = {
    "confusion":
        "The user didn't understand. Reformulate your previous answer more simply: {last_answer}",
    "half_understood":
        "The user partially understood. Ask what specific part was unclear, then clarify: {last_answer}",
    "simplify":
        "The user wants an example. Re-explain using a concrete real-world example.",
    "repeat_all":
        "Repeat your previous answer from the beginning: {last_answer}",
    "repeat_part":
        "Repeat only the last paragraph of your previous answer: {last_answer}",
    "repeat":
        "Repeat your previous answer: {last_answer}",
    "hallucination":
        "You went off topic. Return strictly to the original question and answer only from the document context.",
    "wrong_answer":
        "The user says your answer was wrong. Re-read the context carefully and try again.",
    "giving_up":
        "The user is giving up. Make one final attempt with the simplest possible explanation in 2 sentences maximum.",
    "impatience":
        "The user wants a shorter answer. Respond in maximum 2 sentences, direct and to the point.",
}

def _hybrid_retrieve(question, bm25, db):
    """Vector search (with scores) + BM25 keyword fill, deduplicated."""
    seen, docs = set(), []

    for doc, score in db.similarity_search_with_score(question, k=4):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            doc.metadata["score"] = score
            docs.append(doc)

    for doc in bm25.invoke(question):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            doc.metadata["score"] = None  # BM25 has no vector distance
            docs.append(doc)

    return docs

def stream_answer(question, chat_history=None, intent="none", last_answer="", lang="en"):
    from utils import get_system_prompt

    system_prompt = get_system_prompt(lang)

    # Build augmented question from intent instruction
    tmpl = _INTENT_INSTRUCTIONS.get(intent, "")
    if tmpl:
        # Truncate last_answer to avoid bloating the prompt
        snippet = (last_answer or "")[:500]
        augmented = tmpl.format(last_answer=snippet) + "\n\nOriginal question: " + question
    else:
        augmented = question

    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    raw = db._collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(raw["documents"], raw["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = 4

    unique_docs = _hybrid_retrieve(question, bm25, db)
    context = "\n\n".join(doc.page_content for doc in unique_docs)

    history_section = ""
    if chat_history:
        lines = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in chat_history
        ]
        history_section = "Conversation history:\n" + "\n".join(lines) + "\n\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         system_prompt + "\n\n"
         "Answer based on the provided context and conversation history.\n\n"
         "Context:\n{context}\n\n"
         "{history_section}"),
        ("human", "{question}"),
    ])

    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.3, max_tokens=700)
    stream = (prompt | llm | StrOutputParser()).stream({
        "context": context,
        "question": augmented,
        "history_section": history_section,
    })
    return stream, unique_docs
