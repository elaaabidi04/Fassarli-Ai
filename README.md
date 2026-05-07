# Fassarli AI — Tunisian Darija RAG Chatbot

> *"Fassarli" (فسرلي) means "explain it to me" in Tunisian Darija.*

A multilingual document Q&A chatbot with **native Tunisian Darija support** — the first RAG system built specifically for Tunisian Arabizi. Powered by **Llama 3.3 70B** via NVIDIA NIM with a hybrid BM25 + vector retrieval pipeline and a self-improving feedback loop.

---

## Why This Project Is Different

Most RAG chatbots handle English, French, maybe Arabic. None of them handle Tunisian Darija — a dialect that mixes Arabic, French, and Berber, written in Latin script with numbers replacing missing letters (`3=ع`, `5=خ`, `7=ح`, `9=ق`). Fassarli does.

The system detects when a user writes `mafhemtch chy` (I didn't understand anything) or `5arreeff` (you're hallucinating) and **automatically adapts its response** — reformulating, repeating, giving examples, or acknowledging it went off topic. It does this across **4 languages simultaneously**, switching per message with no manual input.

---

## Darija Intent System

The core innovation: a custom NLP layer that reads Tunisian Darija signals and changes how the model responds in real time. 26 intents, detected by fuzzy matching with priority ordering.

| Priority | Intent | Trigger example | What the model does |
|---|---|---|---|
| p2 | `giving_up` | ayast moi, la3alayna | One final 2-sentence attempt |
| p3 | `hallucination` | 5arreeff, khareef | Returns strictly to document context |
| p3.5 | `pdf_mismatch` | mouch hakka fil pdf | Corrects answer to match source material |
| p4 | `wrong_answer` | mouch hakka, moch kk | Re-reads context and retries |
| p4.5 | `source_check` | mnayn jibt hadha | Cites the specific retrieved chunk |
| p5 | `confusion` | mafhemtch, chnou, zid fahamni | Reformulates more simply |
| p5.2 | `persistent_confusion` | ma fhemtech zeda | Switches strategy: analogy + schema |
| p5.5 | `pdf_reference` | fil pdf 9alek | Confirms or clarifies what the PDF says |
| p5.8 | `pdf_confusion` | 9ritha mafhemtch | Explains the PDF concept step by step |
| p6 | `half_understood` | fhemt ama mouch barcha | Asks what part was unclear |
| p6.5 | `clarification_question` | kifesh exactement | Answers the specific sub-question |
| p6.6 | `definition_question` | chnou ya3ni, 3arrafli | Gives a short definition + Tunisian example |
| p6.8 | `why_question` | 3lech hakka, pourquoi | Explains the rationale + consequence |
| p6.9 | `compare_question` | farq bin X w Y | Defines both, states difference, gives example |
| p7 | `simplify` | a3tini example, het mithal | Re-explains with a concrete Tunisian analogy |
| p7.2 | `consequence_question` | chnou yasser luka | Explains the consequence with a practical example |
| p8 | `repeat_all` | 3awed miloul | Repeats from the beginning |
| p8.5 | `continue_deeper` | zid, akther, go deeper | Adds more detail and advanced nuance |
| p9 | `repeat_part` | lfaza le5ra, 3awidly hethi | Repeats last paragraph only |
| p10 | `repeat` | 3awidly, awedly | Repeats previous answer |
| p11 | `impatience` | na9is mel7dith | Maximum 2-sentence response |
| p12 | `understood` | sa7it, mrgl, jawek behi | Saves exchange to knowledge base |
| p13 | `pushback` | la mouch hakka, ana naaraf | Acknowledges disagreement, addresses the point |
| p14.5 | `closing_thanks` | merci, barka, 3awed mrigla | Warm close, confirms understanding |
| — | `swear` | *(private list)* | 10-second cooldown, stripped from input |
| — | `encouragement` | bahi, barcha mzyan | Continues normally with positive tone |

Intent priority is enforced — if a message contains multiple signals, the highest priority wins. Fuzzy matching handles the vowel-dropping and space-skipping patterns common in real Tunisian typing (`mrgl` = `mrigl` = `mregl`).

---

## /improve Command

Users can suggest corrections directly from the chat without leaving the interface.

```
User types: /improve the example about banking was wrong, should be...
              (also works: /7assin, /7assen, /hassin)
                    ↓
Suggestion saved to "User Suggestions" sheet in darija_nlp_database.xlsx
                    ↓
Admin reviews in Darija Manager → Suggestions tab
                    ↓
Approve → mark validated → Sync to ChromaDB
Reject  → mark rejected with reviewer note
```

The chat input turns purple when a `/improve` command is detected, so users know it's being treated as a correction, not a regular question.

---

## Darija NLP Database

A structured Excel database (`darija_nlp_database.xlsx`) built from real Tunisian social media comments, with an automated classification pipeline:

```
Paste raw Facebook/YouTube comments into Excel
        ↓
Click "Classify New Rows" → Llama 3.3 fills intent, variants, translations
        ↓
Review flagged rows in Darija Manager (password-protected)
        ↓
Click "Sync to ChromaDB" → validated rows pushed to vector DB
        ↓
Chatbot immediately responds better in Darija
```

| Sheet | Purpose |
|---|---|
| Raw Collection | Individual expressions with intent, variants, translations |
| Conversation Threads | Full validated Q&A exchanges for RAG training |
| Intent Signals Reference | Intent → action mapping used by the retriever |
| Export Preview | Shows how data will look in ChromaDB |
| Stats | Collection progress dashboard with targets per intent |
| User Suggestions | Correction suggestions submitted via /improve from the chat |

---

## System Architecture

```
User message
     │
     ▼
┌─────────────┐     ┌──────────────────┐
│ Swear check │────▶│ 10s cooldown     │
└─────────────┘     └──────────────────┘
     │ clean text
     ▼
┌─────────────────┐
│ Language detect │──── Darija signals (fuzzy) → darija
│                 │──── langdetect (conf > 0.85) → fr/ar/en
└─────────────────┘
     │ lang
     ▼
┌─────────────────┐
│ Intent detect   │──── 26 intent categories, priority order (p2–p14.5)
│ (fuzzy match)   │──── normalize → strip spaces → collapse repeats
└─────────────────┘
     │ intent
     ▼
┌──────────────────────────────────────┐
│ Hybrid Retrieval                     │
│  BM25 keyword (40%) +                │
│  NVIDIA embeddings (60%)             │
│  → merged, deduplicated, top-k       │
└──────────────────────────────────────┘
     │ context chunks
     ▼
┌─────────────────────────────────────┐
│ System prompt builder               │
│  → language-specific instructions  │
│  → intent-specific reformulation   │
│  → Darija response style examples  │
└─────────────────────────────────────┘
     │ augmented prompt
     ▼
┌──────────────────────┐
│ Llama 3.3 70B (NIM)  │ streaming
└──────────────────────┘
     │ answer
     ▼
┌──────────────────────────────────────┐
│ Feedback loop                        │
│  thumbs up → save to ChromaDB        │
│  thumbs down → delete from ChromaDB  │
│  /improve → save to User Suggestions │
└──────────────────────────────────────┘
```

---

## Features

- **Multilingual chat** — English, French, Arabic (MSA), Tunisian Darija — detected per message, switches freely
- **26-intent Darija NLP** — fuzzy-matched intent detection with priority ordering, covering confusion, correction, repetition, clarification, and more
- **Hybrid retrieval** — BM25 keyword + NVIDIA semantic embeddings, merged and deduplicated
- **Streaming answers** — token-by-token output via NVIDIA NIM
- **Feedback loop** — thumbs up/down saves or removes Q&A pairs from the knowledge base
- **/improve command** — users submit corrections from the chat; reviewed in Darija Manager before touching ChromaDB
- **Swear word filter** — configurable via `.env`, 10s cooldown, never logged or repeated
- **Persistent chat** — conversation survives page refresh, cleared only on button click
- **Darija Manager** — password-protected CRUD interface for the NLP database + suggestion review
- **Auto-classification** — LLM classifies raw Darija comments into structured intent data
- **Vector DB Explorer** — visualize and inspect ChromaDB embeddings
- **Pinned PDF** — a knowledge document that is always indexed and cannot be deleted

---

## Project Structure

```
Q&A/
├── app.py                         # Main Streamlit app (all pages)
├── retriever.py                   # Hybrid retrieval + streaming answer logic
├── ingest.py                      # PDF ingestion → ChromaDB
├── utils.py                       # Language detection, intent detection, swear filter
├── feedback.py                    # Save / delete validated Q&A exchanges in ChromaDB
├── classify_comments.py           # LLM-based auto-classification of raw Darija comments
├── export_to_rag.py               # Export validated Excel rows → ChromaDB chunks
├── ingest_darija.py               # Ingest darija_chunks.json into ChromaDB
├── suggestions.py                 # Save / load / update user-submitted /improve corrections
├── _compat.py                     # Python 3.14 + urllib3 header encoding patch
├── darija_nlp_database.xlsx       # Tunisian Darija NLP database (6 sheets)
├── tunisian_darija_nlp_guide.pdf  # Pinned knowledge-base PDF (never deleted)
├── chroma_db/                     # Persistent vector store (local, no server needed)
├── requirements.txt               # All dependencies
├── .env                           # API keys and config — never commit this
└── .streamlit/
    └── config.toml                # Theme config
```

---

## Setup

### 1. Prerequisites

- Python 3.11+ (tested on 3.11 and 3.14)
- Free [NVIDIA NIM](https://build.nvidia.com) account

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Everything is in `requirements.txt`. No separate installs needed.

### 3. Configure `.env`

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=nvapi-...
SWEAR_WORDS=word1,word2,word3
DARIJA_CODE=your_secret_password
PINNED_PDF=tunisian_darija_nlp_guide.pdf
```

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | API key from build.nvidia.com |
| `SWEAR_WORDS` | Comma-separated list — no quotes, kept private |
| `DARIJA_CODE` | Password to unlock Darija Manager |
| `PINNED_PDF` | Filename of the always-indexed PDF (leave empty to disable) |

> Never add quotes around `.env` values. Comments must be on their own line starting with `#`.

### 4. Run

```bash
streamlit run app.py
```

---

## Pages

### Q&A Chat
Upload up to 5 PDFs. Ask in any supported language. The pinned PDF is always indexed and cannot be deleted by any user. Use `/improve` (or `/7assin`, `/7assen`, `/hassin`) to suggest a correction to any answer.

### Darija Manager *(password-protected)*
Manage the Raw Collection and Conversation Threads sheets of the NLP database, and review user-submitted corrections.

| Tab | What it does |
|---|---|
| Edit & Validate | Edit intent, aggression, variants inline. Set `validated = yes` to approve. |
| Select & Delete | Click rows to select, delete unvalidated entries permanently. |
| Suggestions | Review /improve corrections submitted from the chat. Approve or reject with a note. |

**Sidebar buttons:**
- **Classify New Rows** — sends unclassified raw comments to Llama 3.3 for auto-classification. Live progress shown in sidebar.
- **Sync to ChromaDB** — exports all `validated = yes` rows into ChromaDB immediately.

### Vector DB Explorer
Inspect indexed chunks, visualize the embedding space, browse document metadata.

---

## Models

| Model | Provider | Purpose |
|---|---|---|
| `meta/llama-3.3-70b-instruct` | NVIDIA NIM | Chat, classification, streaming |
| `nvidia/nv-embedqa-e5-v5` | NVIDIA NIM | Document and query embeddings |

Both free tier via a single `NVIDIA_API_KEY`.

---

## Notes

- Chat history persists in `chat_history.json`. Clear with the **Clear conversation** button.
- Temp PDF files are cleaned up automatically on startup and after every upload.
- `_compat.py` patches a Python 3.14 + urllib3 incompatibility with non-Latin-1 HTTP headers — do not remove it if running Python 3.14.
- ChromaDB runs fully local at `./chroma_db/` — no server, no cloud, no cost.
- `chromadb==0.4.24` requires `numpy<2` — pinned in `requirements.txt` to prevent a NumPy 2.0 incompatibility. Do not upgrade either without testing.
- ChromaDB imports are lazy (loaded only when Sync/Classify buttons are clicked) to keep app startup fast and avoid dependency conflicts.

---

## Built With

`LangChain` · `ChromaDB` · `NVIDIA NIM` · `Llama 3.3 70B` · `Streamlit` · `BM25` · `rapidfuzz` · `openpyxl` · `langdetect`

## License
MIT © 2026 Elaa Abidi
