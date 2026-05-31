"""Convert ISKCON Bhagavad-gita PDF diacritics to standard IAST and Devanagari."""

import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# The ISKCON PDF embeds a custom Type 1 font whose glyphs map to Latin-1
# supplement characters when extracted with pypdf. This table maps those
# characters to standard Unicode IAST.
ISKCON_PDF_TO_IAST = str.maketrans(
    {
        "\u00e5": "\u1e5b",  # å -> ṛ
        "\u00e4": "\u0101",  # ä -> ā
        "\u00f1": "\u1e63",  # ñ -> ṣ
        "\u00f6": "\u1e6d",  # ö -> ṭ
        "\u00eb": "\u1e47",  # ë -> ṇ
        "\u00f2": "\u1e0d",  # ò -> ḍ
        "\u00ef": "\u00f1",  # ï -> ñ
        "\u00e9": "\u012b",  # é -> ī
        "\u00fc": "\u016b",  # ü -> ū
        "\u00e0": "\u1e41",  # à -> ṁ
        "\u00ec": "\u1e41",  # ì -> ṁ
        "\u00f9": "\u1e25",  # ù -> ḥ
        "\u00e7": "\u015b",  # ç -> ś
    }
)


def iskcon_pdf_to_iast(text: str) -> str:
    """Convert ISKCON PDF roman diacritics to standard Unicode IAST."""
    return text.translate(ISKCON_PDF_TO_IAST)


# Multi-character replacements first (order matters for overlapping patterns).
_IAST_MULTI_TO_PLAIN = (
    ("ṣ", "sh"),
    ("Ṣ", "Sh"),
    ("ś", "sh"),
    ("Ś", "Sh"),
    ("ṛ", "ri"),
    ("Ṝ", "Ri"),
)

# Single-character IAST diacritics → ASCII letters TTS can read naturally.
_IAST_SINGLE_TO_PLAIN = str.maketrans(
    {
        "ā": "a",
        "ī": "i",
        "ū": "u",
        "ē": "e",
        "ō": "o",
        "Ā": "A",
        "Ī": "I",
        "Ū": "U",
        "Ē": "E",
        "Ō": "O",
        "ḥ": "h",
        "Ḥ": "H",
        "ṁ": "m",
        "ṃ": "m",
        "ṅ": "ng",
        "ñ": "n",
        "Ñ": "N",
        "ṇ": "n",
        "ḍ": "d",
        "ṭ": "t",
        "Ḍ": "D",
        "Ṭ": "T",
        "ç": "sh",
    }
)


def iast_to_plain_english(text: str) -> str:
    """Strip IAST diacritics so English TTS reads names like Sanjaya, Pandu."""
    if not text:
        return text
    for src, dst in _IAST_MULTI_TO_PLAIN:
        text = text.replace(src, dst)
    text = text.translate(_IAST_SINGLE_TO_PLAIN)
    return text


def extract_iast_block(verse_text: str) -> str:
    """Extract the roman transliteration block from a verse section."""
    blocks = extract_iast_blocks(verse_text)
    if not blocks:
        return ""
    return "\n".join(text for _, text in blocks)


def extract_iast_blocks(verse_text: str) -> list[tuple[int, str]]:
    """Extract one or more transliteration blocks (handles TEXTS 3-4 style)."""
    pattern = re.compile(
        r"\)\)\s*(\d+)\s*\)\)\s*\n(.*?)(?=\)\)\s*\d+\s*\)\)|\nSYNONYMS)",
        re.DOTALL,
    )
    found = pattern.findall(verse_text)
    if found:
        return [(int(num), text.strip()) for num, text in found]
    match = re.search(r"\)\)\s*\d+\s*\)\)\s*\n(.+?)\nSYNONYMS", verse_text, re.DOTALL)
    if match:
        return [(0, match.group(1).strip())]
    return []


def iast_to_devanagari(iast_text: str) -> str:
    """Convert IAST verse text to Devanagari suitable for gTTS."""
    plain = iast_text.replace("-", "")
    plain = re.sub(r"\s+", " ", plain.replace("\n", " ")).strip()
    if not plain:
        return ""
    return transliterate(plain, sanscript.IAST, sanscript.DEVANAGARI)


def parse_verse_sanskrit(verse_body: str) -> tuple[str, str]:
    """Return (iast, devanagari) parsed from a verse body."""
    raw_iast = extract_iast_block(verse_body)
    iast = iskcon_pdf_to_iast(raw_iast)
    devanagari = iast_to_devanagari(iast)
    return iast, devanagari
