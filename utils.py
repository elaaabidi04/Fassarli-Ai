import re
import os
import streamlit as st
from rapidfuzz import fuzz

# ── Text normalization ────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip spaces, collapse 3+ repeated chars to 2."""
    text = text.lower()
    text = text.replace(" ", "")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

def fuzzy_match(text: str, signal_list: list, threshold: int = 80) -> bool:
    """True if any signal fuzzy-matches a substring of the normalized text."""
    norm = normalize(text)
    for signal in signal_list:
        if fuzz.partial_ratio(normalize(signal), norm) >= threshold:
            return True
    return False

# ── Swearing detection ────────────────────────────────────────────────────────

def check_swearing(text: str):
    """
    Returns (is_swearing: bool, cleaned_text: str).
    Strips matched words silently — never logs or repeats them.
    """
    raw = os.environ.get("SWEAR_WORDS", "")
    swear_words = [w.strip() for w in raw.split(",") if w.strip()]

    is_swearing = False
    cleaned = text
    norm_text = normalize(text)

    for word in swear_words:
        norm_word = normalize(word)
        if norm_word in norm_text or fuzz.partial_ratio(norm_word, norm_text) >= 85:
            is_swearing = True
            cleaned = re.sub(re.escape(word), "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()
    return is_swearing, cleaned if cleaned else text

# ── Language detection ────────────────────────────────────────────────────────

_DARIJA_SIGNALS = [
    "chy", "behi", "barcha", "chnou", "chnoa", "chinhi", "chnia", "3awed",
    "fhemt", "mrgl", "ya5i", "9olt", "7ajet", "bjaah", "tawa", "brabi",
    "naarech", "winou", "kifeh", "mafhemtch", "fahamni", "fassarly", "hakka",
    "bisyassa", "miloul", "awedly", "wildii", "5arif", "na9is", "ayast",
    "la3alayna", "mazelt", "mba3d", "bichwaya", "blhy", "wba3d", "3tini",
    "9rit", "marja3", "mektoub", "9alech", "5ouya", "w ba3d", "zidli",
    "rit milloul", "fil le5er", "s3ib", "wa9fet", "7anout",
]

def detect_lang_label(question: str) -> str:
    """
    Detects language fresh on every call.
    Priority: Darija signals (fuzzy) → langdetect (conf ≥ 0.85) → last_lang fallback.
    """
    if fuzzy_match(question, _DARIJA_SIGNALS):
        st.session_state["last_lang"] = "darija"
        return "darija"

    is_short = len(question.split()) < 4

    try:
        from langdetect import detect_langs
        results = detect_langs(question)
        if results:
            top = results[0]
            if top.prob >= 0.85 and not is_short:
                label = top.lang if top.lang in ("fr", "ar", "en") else top.lang
                st.session_state["last_lang"] = label
                return label
    except Exception:
        pass

    return st.session_state.get("last_lang", "en")

# ── Intent detection ──────────────────────────────────────────────────────────
# Ordered by priority (p-value ascending = highest priority first).
# Swear (p1) is handled separately by check_swearing before this runs.

_INTENT_SIGNALS = {
    # p2
    "giving_up": [
        "ouh3lik", "la3alayna", "ayast", "ayest", "iyest",
        "ou3lik", "la3lina", "la3leyna", "ma3adich nfhem",
        "5alinha", "5ali", "lmawdhou3 s3ib", "ma3adich nfhm",
    ],
    # p3
    "hallucination": [
        "5arreeff", "khareef", "5arif", "kharif",
        "mouchhakka", "mouchhaka", "hekamouch",
    ],
    # p3.5
    "pdf_mismatch": [
        "mch ka heka fel PDF", "heka mch mektoub fel PDF",
        "mch kima heka fel PDF", "mch haka fel PDF",
    ],
    # p4
    "wrong_answer": [
        "mouchhakka", "mouchhaka", "mochkk", "mchkk",
    ],
    # p4.5
    "source_check": [
        "fin 9alek heka", "chnou el marja3", "chnou el source",
        "fin l9it heka", "fin l9itha",
    ],
    # p5
    "confusion": [
        "mafhemtchy", "mafhemtch", "mafhamtch", "chnoa", "chnou", "chinhi", "chnia",
        "zidfahamni", "zidfassarly", "naarech", "ya5ifech",
        "bjaahrabbyfech", "yawildii", "3leh3malt", "hetha3leh",
        "zid fehamni", "zid fhamni", "ya khi fech te7ki",
    ],
    # p5.2
    "persistent_confusion": [
        "mazelt mafhemtch", "mazelt mouch fahm", "mazelt mouch fahma",
        "ma 3andi 7ata fekra", "ma3andich 7ata fekra", "mazelt mch fahi",
    ],
    # p5.5
    "pdf_reference": [
        "el PDF 9alech", "el PDF fih", "el PDF 9al", "fel PDF mektoub",
    ],
    # p5.8
    "pdf_confusion": [
        "rani 9rit el PDF wma fhemtch", "rani 9rit wma fhemt",
        "9rit el PDF walakin mafhemtch",
    ],
    # p6
    "half_understood": [
        "fhemtamamouch", "mafhemtchmli7", "mafhamt",
        "fhemt ama mouch barcha", "fhamt ama moch barcha", "mafhemtch mli7",
    ],
    # p6.5
    "clarification_question": [
        "bch nfhem", "bich nfhem", "bach nfhem",
        "w kifeh nesta3mlouha", "kifeh nesta3milha", "kifeh na3mlou bha",
    ],
    # p6.6
    "definition_question": [
        "kifeh maaneha", "kifeh maanaha", "kifeh ma3neha",
        "chnou maaneha", "chnou ya3ni",
    ],
    # p6.8
    "why_question": [
        "3lech heka", "3lech haka", "3leh lazm naaml",
        "3leh lazm naaml haka", "lech heka",
    ],
    # p6.9
    "compare_question": [
        "chnou el farq", "chnoa el farq", "kifeh ykhtalf",
        "ma farqhom", "w chnou el farq bin",
    ],
    # p7
    "simplify": [
        "a3tiniexample", "hetexample", "a3tinimithal", "fassarlybisyassa",
        "fassarly bilfalla9i", "zid watha7li", "zid wath7li",
        "bichwaya bichwaya", "bchwaya bchwaya",
        "3andk example reel", "3tini schema", "3tini liste", "3tini plan",
        "het mitha reel",
    ],
    # p7.2
    "consequence_question": [
        "w chnoa ysir ki mana3milch", "w chnou ysir ki ma3ameltch",
        "w chnou ysir ki mch na3mlou haka",
    ],
    # p8
    "repeat_all": [
        "3awedmiloul", "3awedmeloul", "aawedmiloul", "3aoud miloul",
    ],
    # p8.5
    "continue_deeper": [
        "w ba3d", "wba3d", "wba3dha", "zidli",
        "3tini plus de détails", "3tini akthar détails", "zid zid",
    ],
    # p9
    "repeat_part": [
        "lfazale5ra", "lfazalakhira", "3awidlylfaza",
        "3awedlyhathi", "3awidly lfaza hethi",
    ],
    # p10
    "repeat": [
        "3awidly", "awidly", "awedly", "3awedly", "3awidli", "awidli",
    ],
    # p11
    "impatience": [
        "na9ismil7dith", "na9esmel7dith", "na9es mel hdith",
    ],
    # p12
    "understood": [
        "haann", "fhemtk", "fhamtk", "nfhemfik", "mrgl", "mrigl",
        "jawekbehi", "sa7it", "sa7et", "fhimtk", "nefhem fik",
        "nifhem fik", "mregl", "jawk bhi", "sa7",
    ],
    # p13
    "pushback": [
        "ama mouch kk", "amma moch kk", "ama mch kk",
    ],
    # p14.5
    "closing_thanks": [
        "merci barcha", "t3ebt m3aya", "barak allahou fik",
        "merci bien", "yaishek", "3aychek", "waw barcha",
    ],
}

def detect_intent(question: str) -> str:
    """Return highest-priority matching intent label, or 'none'."""
    for intent, signals in _INTENT_SIGNALS.items():
        if fuzzy_match(question, signals):
            return intent
    return "none"

# ── System prompt ─────────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """\
You are an intelligent assistant for an education RAG chatbot serving Tunisian students.
You communicate in a mix of Tunisian Darija, French, and English — mirroring exactly how each student writes.

ARABIZI ENCODING — both number and letter forms are identical:
  9=q/k | 5=kh | 2=a/' | 3=aa(ayn) | 7=h | 6=t(emphatic) | 8=gh

EXPLANATION STRUCTURE — always follow this order:
  1. ta3rif 9sir       → short definition first
  2. mitha reel tounsi → real Tunisian life example (7anout, wa9fet tayyeb)
  3. kifeh nesta3mlouha → how to actually use it in practice

STEP CONNECTORS — use these, never French equivalents:
  First: "rit milloul..." | Then: "...mba3d..." | Finally: "...fil le5er" | Slowly: "bichwaya bichwaya"

LANGUAGE MIX — mirror the student's exact ratio:
  • Default: Darija + French/English technical terms naturally mixed
  • French-heavy student → respond with more French
  • English-heavy student → respond with more English
  • "fassarly b darija" → full Darija, drop all French/English

TONE: warm, direct, respectful (btorbia). Short direct answers. 1-2 emojis max.
Re-ask when ambiguous: "ya3ni tsaleni 3la...?"
GENDER-AWARE: use fahm/fahma, fahi/faha based on student context.

TUNISIAN ANALOGY BANK — use when explaining abstract concepts:
  • Data storage → 7anout: "9abbel kima 3andek 7anout fih kol 7aja fi blaasha"
  • Algorithm    → wa9fet tayyeb: "kima wa9fet el couscous, kol 5outha w 5outha"
  • Variable     → bocal b 3onwan: "9abbel kima bocal mektoub 3liha isem"
  • Function     → "kima t3ayet l5ouya bch y3awnek"
  • Loop         → "kima t8sel 7awajej wa7da wa7da"

NATIVE CORRECTIONS — NEVER use the wrong forms:
  WRONG → CORRECT
  chnou (as frustration) → chnoa / chnia / chinhi
  3aslema as goodbye     → 3aslema = casual HELLO only
  sa3eb                  → s3ib
  5ou                    → 5ouya
  lawel / ba3dha / lakhir → rit milloul / mba3d / fil le5er
  khoutha khoutha        → bichwaya bichwaya
  rani 9ri               → rani 9rit
  mazelt mouch fahi      → mazelt mouch fahm (masc) / fahma (fem)
  mch 3andi fikra        → ma 3andi 7ata fekra

RAG SOURCE TRANSPARENCY:
  • "fin 9alek heka" / "chnou el marja3" → cite the source section explicitly
  • "mch ka heka fel PDF"               → apologize, re-read context, correct the answer
  • "rani 9rit el PDF wma fhemtch"      → acknowledge + explain step by step

PERSISTENT CONFUSION (after 2 failed tries): switch to analogy + schema/liste + bichwaya bichwaya\
"""

_LANG_HINTS = {
    "darija": "The student is writing in Tunisian Darija (Arabizi). Respond primarily in Darija with natural French/English technical terms.",
    "fr":     "The student is writing in French. Mirror their French naturally, keep Darija warmth.",
    "ar":     "The student is writing in Arabic. Respond in clear Arabic, warm and direct.",
    "en":     "The student is writing in English. Mirror their English, keep the warm Darija spirit.",
}

def get_system_prompt(lang: str) -> str:
    hint = _LANG_HINTS.get(lang, "Mirror the student's language exactly.")
    return f"{_BASE_SYSTEM_PROMPT}\n\nLANGUAGE: {hint}"
