import chromadb
import os
import re
import shutil
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pathlib import Path
from agents.sanskrit_translation import (
    extract_iast_blocks,
    iast_to_devanagari,
    iskcon_pdf_to_iast,
    parse_verse_sanskrit,
)
from agents.tts import speak_english,speak_sanskrit


PDF_PATH = 'Bhagavad-gita_As_It_Is english.pdf'

# Load & Chunk PDF
def load_pdf(pdf_path, chunk_size=500, overlap=50):
    reader = PdfReader(pdf_path)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    return strip_copyright(text)

# ISKCON PDF embeds this notice ~1000×; it also garbles when transliterated to Devanagari.
COPYRIGHT_LINE_RE = re.compile(
    r"Copyright\s*(?:©|\u00a9|\u00ae)?\s*1998\s*"
    r"The\s*Bhaktivedanta\s*Book\s*Trust\s*Int['\u2019]?l\.?\s*"
    r"All\s*Rights\s*Reserved\.?",
    re.IGNORECASE,
)
COPYRIGHT_INLINE_RE = re.compile(
    r"Copyright[^\n]{0,160}?Bhaktivedanta[^\n]{0,120}?Reserved\.?",
    re.IGNORECASE,
)
# Devanagari transliteration of the same notice (from PDF font / gTTS path).
COPYRIGHT_DEVANAGARI_RE = re.compile(
    r"[\u0900-\u097F\s©©]{0,40}?१९९८[\u0900-\u097F\s©©'a-zA-Z.]{0,220}?"
    r"(?:रिघ्त्स्|रेषर्वेद|भक्तिवेदन्त|Rights|Reserved)[\u0900-\u097F\s©©'a-zA-Z.]*",
    re.IGNORECASE,
)


def strip_copyright(text: str) -> str:
    """Remove Bhaktivedanta copyright lines (English and garbled Devanagari)."""
    if not text:
        return text
    text = COPYRIGHT_LINE_RE.sub("", text)
    text = COPYRIGHT_INLINE_RE.sub("", text)
    text = COPYRIGHT_DEVANAGARI_RE.sub("", text)
    text = re.sub(
        r"^[^\n]*(?:Book Trust|All Rights Reserved)[^\n]*\n?",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return text


def clean(text: str) -> str:
    text = strip_copyright(text)
    text = re.sub(r"\s{3,}", " ", text)
    return text.strip()

# Extract Section
def extract_section(text: str, start_kw: str, end_kw: list[str]) -> str:
    start_match = re.search(rf'\b{start_kw}\b', text)
    if not start_match:
        return ''
    start = start_match.end()
    end = len(text)
    for kw in end_kw:
        m = re.search(rf'\b{kw}\b', text[start:])
        if m:
            end = min(end, start + m.start())
    return text[start:end].strip()

# TEXT 5 | TEXTS 3-4 | TEXTS 8–12 (PDF uses TEXTS + en-dash for combined ślokas)
TEXT_HEADER_RE = re.compile(
    r"TEXTS?\s+(\d+)(?:\s*[-–—\u2013\u2014]\s*(\d+))?\s*",
    re.IGNORECASE,
)


def _build_verse_record(
    chapter: int,
    verse_num: int,
    iast: str,
    devanagiri: str,
    translation: str,
    synonyms: str,
    purport: str,
    verse_end: int | None = None,
) -> dict:
    label = (
        f"Chapter {chapter}, Verses {verse_num}–{verse_end}."
        if verse_end and verse_end != verse_num
        else f"Chapter {chapter}, Verse {verse_num}."
    )
    return {
        "id": f"ch{chapter}_v{verse_num}",
        "chapter": chapter,
        "verse": verse_num,
        "verse_end": verse_end or verse_num,
        "iast": iast,
        "devanagiri": devanagiri,
        "translation": translation,
        "synonyms": synonyms,
        "purport": purport,
        "embed_text": f"{label}{devanagiri} {translation}",
    }


def parse_gita(full_text: str) -> list[dict]:
    verses = []
    chapter_splits = re.split(r"(-\s*CHAPTER\s+(\d+)\s*-)", full_text)

    for i in range(1, len(chapter_splits), 3):
        current_chapter = int(chapter_splits[i + 1])
        chapter_body = chapter_splits[i + 2] if i + 2 < len(chapter_splits) else ""

        parts = TEXT_HEADER_RE.split(chapter_body)
        j = 1
        while j + 2 <= len(parts):
            verse_start = int(parts[j])
            verse_end = int(parts[j + 1]) if parts[j + 1] else verse_start
            verse_body = strip_copyright(parts[j + 2])
            j += 3

            verse_nums = list(range(verse_start, verse_end + 1))
            translation = clean(
                iskcon_pdf_to_iast(
                    extract_section(verse_body, "TRANSLATION", ["PURPORT", "TEXT", "TEXTS"])
                )
            )
            synonyms = clean(
                iskcon_pdf_to_iast(
                    extract_section(verse_body, "SYNONYMS", ["TRANSLATION", "PURPORT"])
                )
            )
            purport = clean(
                iskcon_pdf_to_iast(extract_section(verse_body, "PURPORT", ["TEXT", "TEXTS"]))
            )

            raw_blocks = extract_iast_blocks(verse_body)
            per_verse_sanskrit: dict[int, tuple[str, str]] = {}

            if len(raw_blocks) >= len(verse_nums):
                for idx, vnum in enumerate(verse_nums):
                    block_num, raw = raw_blocks[idx]
                    vkey = block_num if block_num else vnum
                    iast = clean(iskcon_pdf_to_iast(strip_copyright(raw)))
                    per_verse_sanskrit[vkey] = (iast, clean(iast_to_devanagari(iast)))
            else:
                iast, devanagiri = parse_verse_sanskrit(verse_body)
                iast, devanagiri = clean(iast), clean(devanagiri)
                for vnum in verse_nums:
                    per_verse_sanskrit[vnum] = (iast, devanagiri)

            shared_end = verse_end if len(verse_nums) > 1 else None
            for vnum in verse_nums:
                iast, devanagiri = per_verse_sanskrit.get(
                    vnum, per_verse_sanskrit.get(verse_nums[0], ("", ""))
                )
                verses.append(
                    _build_verse_record(
                        current_chapter,
                        vnum,
                        iast,
                        devanagiri,
                        translation,
                        synonyms,
                        purport,
                        verse_end=shared_end,
                    )
                )
    return verses

# Set up ChromaDB
STORE_PATH = "gita_store"
INDEX_VERSION = "5"

if os.getenv("GITA_REINDEX", "").lower() in ("1", "true", "yes"):
    shutil.rmtree(STORE_PATH, ignore_errors=True)

client = chromadb.PersistentClient(path=STORE_PATH)

encoder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

collection_name = "gita"
existing_collection = [coll.name for coll in client.list_collections()]
print(existing_collection)
if collection_name not in existing_collection:
    collection = client.create_collection(
        name="gita",
        embedding_function=encoder,
        metadata={"hnsw:space": "cosine", "index_version": INDEX_VERSION},
    )

collection = client.get_or_create_collection(collection_name)

needs_reindex = (
    collection.count() == 0
    or collection.metadata.get("index_version") != INDEX_VERSION
)

if needs_reindex:
    if collection.count() > 0:
        client.delete_collection(collection_name)
        collection = client.create_collection(
            name="gita",
            embedding_function=encoder,
            metadata={"hnsw:space": "cosine", "index_version": INDEX_VERSION},
        )
    print("Indexing PDF")
    full_text = load_pdf(PDF_PATH)
    verses = parse_gita(full_text)
    for i in range(0, len(verses), 100):
        batch = verses[i:i+100]
        collection.add(
            documents = [v['embed_text'] for v in batch],
            ids = [v['id'] for v in batch],
            metadatas = [{
                'chapter' : v['chapter'],
                'verse' : v['verse'],
                'verse_end' : v.get('verse_end', v['verse']),
                'iast' : v['iast'],
                'devanagiri' : v['devanagiri'],
                'translation' : v['translation'],
                'synonyms' : v['synonyms'],
                'purport' : v['purport'],
            } for v in batch]
        )
    print(f"Done {len(verses)}")
else:
    print(f"Existing collection contains {collection.count()} chunks")


def _clean_verse_meta(meta: dict) -> dict:
    """Strip copyright from stored fields (covers older indexes)."""
    return {
        k: clean(v) if isinstance(v, str) else v
        for k, v in meta.items()
    }


def get_verse(collection, chapter: int, verse: int) -> dict | None:
    result = collection.get(
        where={"$and": [
            {"chapter": {"$eq": chapter}},
            {"verse": {"$eq": verse}},
        ]}
    )
    if result["metadatas"]:
        return _clean_verse_meta(result["metadatas"][0])
    return None

def format_synonyms(synonyms: str) -> str:
    """Collapse synonym glosses to one line (word—meaning; ...)."""
    return re.sub(r"\s*\n\s*", " ", synonyms).strip()


def get_chapter_verses(collection, chapter: int) -> list[dict]:
    """All verses in a chapter, in ascending order."""
    catalog = get_catalog(collection)
    verses = []
    for v in catalog.get(chapter, []):
        data = get_verse(collection, chapter, v)
        if data:
            verses.append(data)
    return verses


def format_verse_markdown(data: dict, show_translation: bool = True) -> str:
    """Devanagari and English translation only — for chat and sidebar."""
    if not data:
        return "_Verse not found in the Gita store._"
    ch, v = data["chapter"], data["verse"]
    v_end = data.get("verse_end", v)
    title = (
        f"### Chapter {ch}, Verses {v}–{v_end}"
        if v_end and int(v_end) > int(v)
        else f"### Chapter {ch}, Verse {v}"
    )
    devanagiri = clean(data.get("devanagiri") or "")
    translation = clean(data.get("translation") or "")
    lines = [title, "", "**Devanagari**", "", devanagiri or "_—_"]
    if show_translation and translation:
        lines += ["", "**English translation**", "", translation]
    return "\n".join(lines)


def recite_verse(data: dict, with_translation: bool = True) -> None:
    """Autoplay Sanskrit; add English when translation is on."""
    if not data:
        return
    if data.get("devanagiri"):
        speak_sanskrit(data["devanagiri"])
    if with_translation and data.get("translation"):
        speak_english(data["translation"])


def semantic_search(collection, query: str, n_results: int = 3) -> list[dict]:
    """Find verses by meaning or quoted text when chapter/verse are unknown."""
    if not query or not query.strip():
        return []
    results = collection.query(query_texts=[query.strip()], n_results=n_results)
    if not results.get("metadatas") or not results["metadatas"][0]:
        return []
    return [_clean_verse_meta(m) for m in results["metadatas"][0]]


def format_search_results(matches: list[dict], show_translation: bool = True) -> str:
    if not matches:
        return "No matching verses found."
    blocks = [format_verse_markdown(m, show_translation) for m in matches]
    return "\n\n---\n\n".join(blocks)


def get_catalog(collection) -> dict[int, list[int]]:
    """Chapter -> sorted verse numbers available in the vector store."""
    result = collection.get(include=["metadatas"])
    catalog: dict[int, list[int]] = {}
    for meta in result.get("metadatas") or []:
        ch, v = int(meta["chapter"]), int(meta["verse"])
        catalog.setdefault(ch, []).append(v)
    for ch in catalog:
        catalog[ch] = sorted(set(catalog[ch]))
    return dict(sorted(catalog.items()))

