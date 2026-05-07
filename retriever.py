import requests.exceptions

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class NvidiaConnectionError(RuntimeError):
    """Raised when the NVIDIA API is unreachable."""

_INTENT_INSTRUCTIONS = {
    # p2
    "giving_up":
        "The student is giving up. Acknowledge their frustration warmly, then make ONE final attempt "
        "in maximum 2 sentences using the simplest possible explanation. Previous answer: {last_answer}",
    # p3
    "hallucination":
        "The student says you went off-topic or hallucinated. Apologize briefly, then return strictly "
        "to the original question using only what the document context says.",
    # p3.5
    "pdf_mismatch":
        "The student says your answer doesn't match the PDF. Apologize, re-read the context carefully, "
        "and correct your answer to align with the source material.",
    # p4
    "wrong_answer":
        "The student says your answer was wrong. Re-read the context carefully and retry with a corrected answer.",
    # p4.5
    "source_check":
        "The student is asking for your source. Cite the specific document section or chunk you retrieved "
        "this information from. Be transparent about what the RAG context says.",
    # p5
    "confusion":
        "The student didn't understand. Ask for confirmation of what was unclear, then offer to reformulate "
        "more simply. Previous answer: {last_answer}",
    # p5.2
    "persistent_confusion":
        "The student still doesn't understand after multiple tries. Switch strategy completely: "
        "use a Tunisian real-life analogy (7anout, wa9fet tayyeb, bocal), then give a schema or liste, "
        "then explain bichwaya bichwaya. Previous answer: {last_answer}",
    # p5.5
    "pdf_reference":
        "The student is referencing what the PDF says. Confirm or clarify what the retrieved context says "
        "about this. Be specific about the source.",
    # p5.8
    "pdf_confusion":
        "The student read the PDF but didn't understand it. Acknowledge that they read it, "
        "then explain the concept step by step in simpler terms.",
    # p6
    "half_understood":
        "The student partially understood. Ask what specific part was unclear, then clarify only that part. "
        "Previous answer: {last_answer}",
    # p6.5
    "clarification_question":
        "The student is asking a specific sub-question about how something works. "
        "Answer that specific sub-question directly before continuing.",
    # p6.6
    "definition_question":
        "The student wants a definition. Give a short definition first (ta3rif 9sir), "
        "then a real Tunisian example (mitha reel tounsi).",
    # p6.8
    "why_question":
        "The student is asking WHY. Explain the reason or rationale behind the concept clearly, "
        "then give a practical consequence of not following it.",
    # p6.9
    "compare_question":
        "The student wants a comparison. Give: definition of X, definition of Y, "
        "the key difference between them, then a concrete example.",
    # p7
    "simplify":
        "The student wants a simpler explanation or a real example. Re-explain using "
        "a concrete Tunisian real-life analogy (7anout, wa9fet tayyeb, bocal b 3onwan).",
    # p7.2
    "consequence_question":
        "The student is asking what happens if they don't apply the concept. "
        "Explain the consequence clearly with a practical example.",
    # p8
    "repeat_all":
        "The student wants the full answer repeated from the beginning. "
        "Repeat your previous answer completely: {last_answer}",
    # p8.5
    "continue_deeper":
        "The student wants to go deeper. Continue the explanation — add more detail, "
        "sub-cases, or advanced nuance about the topic.",
    # p9
    "repeat_part":
        "The student wants only the last part repeated. "
        "Repeat only the last paragraph of your previous answer: {last_answer}",
    # p10
    "repeat":
        "Repeat your previous answer: {last_answer}",
    # p11
    "impatience":
        "The student is impatient. Respond in MAXIMUM 2 sentences, direct and to the point. No intro.",
    # p12 — no override, proceed normally
    # p13
    "pushback":
        "The student is pushing back on your answer. Acknowledge their disagreement respectfully, "
        "then address the specific point they're contesting. Previous answer: {last_answer}",
    # p14.5
    "closing_thanks":
        "The student is closing the conversation with thanks. Respond warmly, confirm their understanding, "
        "and offer to continue if they have more questions.",
}

_NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

def _hybrid_retrieve(question, bm25, db):
    """Vector search (with scores) + BM25 keyword fill, deduplicated."""
    seen, docs = set(), []

    try:
        vector_results = db.similarity_search_with_score(question, k=4)
    except _NETWORK_ERRORS as e:
        raise NvidiaConnectionError(str(e)) from e

    for doc, score in vector_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            doc.metadata["score"] = score
            docs.append(doc)

    for doc in bm25.invoke(question):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            doc.metadata["score"] = None
            docs.append(doc)

    return docs

def stream_answer(question, chat_history=None, intent="none", last_answer="", lang="en"):
    from utils import get_system_prompt

    system_prompt = get_system_prompt(lang)

    tmpl = _INTENT_INSTRUCTIONS.get(intent, "")
    if tmpl:
        snippet = (last_answer or "")[:500]
        augmented = tmpl.format(last_answer=snippet) + "\n\nOriginal question: " + question
    else:
        augmented = question

    try:
        embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except _NETWORK_ERRORS as e:
        raise NvidiaConnectionError(str(e)) from e

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

    try:
        llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.3, max_tokens=700)
        stream = (prompt | llm | StrOutputParser()).stream({
            "context": context,
            "question": augmented,
            "history_section": history_section,
        })
    except _NETWORK_ERRORS as e:
        raise NvidiaConnectionError(str(e)) from e

    return stream, unique_docs
