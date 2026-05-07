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
    Reads SWEAR_WORDS from .env at call time (comma-separated).
    Strips matched words silently — never logs or repeats them.
    """
    raw = os.environ.get("SWEAR_WORDS", "")
    swear_words = [w.strip() for w in raw.split(",") if w.strip()]

    is_swearing = False
    cleaned = text
    norm_text = normalize(text)

    for word in swear_words:
        norm_word = normalize(word)
        # Direct substring check first (catches exact matches regardless of length),
        # fuzzy fallback handles typos in longer words.
        if norm_word in norm_text or fuzz.partial_ratio(norm_word, norm_text) >= 85:
            is_swearing = True
            cleaned = re.sub(re.escape(word), "", cleaned, flags=re.IGNORECASE)

    cleaned = cleaned.strip()
    return is_swearing, cleaned if cleaned else text

# ── Language detection ────────────────────────────────────────────────────────

_DARIJA_SIGNALS = [
    "chy", "behi", "barcha", "chnou", "chnoa", "3awed", "fhemt", "mrgl",
    "ya5i", "9olt", "7ajet", "bjaah", "tawa", "brabi", "naarech", "winou",
    "kifeh", "mafhemtch", "fahamni", "fassarly", "hakka", "bisyassa", "miloul",
    "awedly", "wildii", "5arif", "na9is", "ayast", "la3alayna",
]

def detect_lang_label(question: str) -> str:
    """
    Detects language fresh on every call (no locked state between messages).
    Priority: Darija signals → langdetect (conf ≥ 0.85) → last_lang fallback.
    Short messages (< 4 words) with low confidence inherit last_lang.
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

    # Fallback — inherit last known language
    return st.session_state.get("last_lang", "en")

# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_SIGNALS = {
    "giving_up":       ["ouh3lik", "la3alayna", "ayast", "ayest", "iyest"],
    "hallucination":   ["5arreeff", "khareef", "5arif", "mouchhakka", "mouchhaka", "hekamouch"],
    "wrong_answer":    ["mouchhakka", "mouchhaka", "mochkk", "mchkk"],
    "confusion":       ["mafhemtchy", "mafhemtch", "chnoa", "chnou", "zidfahamni",
                        "zidfassarly", "naarech", "ya5ifech", "bjaahrabbyfech", "yawildii"],
    "half_understood": ["fhemtamamouch", "mafhemtchmli7", "mafhamt"],
    "simplify":        ["a3tiniexample", "hetexample", "a3tinimithal", "fassarlybisyassa"],
    "repeat_all":      ["3awedmiloul", "3awedmeloul", "aawedmiloul"],
    "repeat_part":     ["lfazale5ra", "lfazalakhira", "3awidlylfaza", "3awedlyhathi"],
    "repeat":          ["3awidly", "awidly", "awedly", "3awedly"],
    "impatience":      ["na9ismil7dith", "na9esmel7dith"],
    "understood":      ["haann", "fhemtk", "fhamtk", "nfhemfik", "mrgl", "mrigl",
                        "jawekbehi", "sa7it", "sa7et"],
}

def detect_intent(question: str) -> str:
    """Return highest-priority matching intent label, or 'none'."""
    for intent, signals in _INTENT_SIGNALS.items():
        if fuzzy_match(question, signals):
            return intent
    return "none"

# ── System prompt builder ─────────────────────────────────────────────────────

def get_system_prompt(lang: str) -> str:
    if lang == "darija":
        return (
            "You MUST respond ONLY in Tunisian Darija written in Arabizi (Latin + numbers). "
            "You are Tunisian, NOT Algerian. Never use Algerian words. "
            "\n\nTunisian vocabulary to USE: behi, barcha, ya5i, w ali, hedheka, chnou, bch, "
            "mte3, 9olt, 7ajet, tawa, brabi, ken, m3a, fi, el, winou, kifeh, 3awed, fhemt, "
            "sa7it, mrigl, ma3lich, nchallah, yezzi, 9rib. "
            "\n\nAlgerian words to NEVER use: wach, daba, bezzaf, khoya, rabi (as greeting), "
            "mzyan, labas (standalone), ntiya, ana, lahne, inty, kima, waqila, makanch, sir, "
            "hak, 3lach, ki, yaaser, goltlek, yak. "
            "\n\nMix French naturally as Tunisians do (mte3, concept, directement, franchement). "
            "Use 3=ع 5=خ 7=ح 9=ق in words. Stay warm and casual. "
            "Example: 'rit houa, bch tfhem barcha...'"
        )
    if lang == "fr":
        return "Réponds en français. Sois clair et concis."
    if lang == "ar":
        return "أجب باللغة العربية الفصحى. كن واضحاً ودقيقاً."
    return "Respond in English. Be clear and concise."
