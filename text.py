import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from rag import (
    collection,
    format_search_results,
    format_verse_markdown,
    get_chapter_verses,
    get_verse,
    recite_verse,
    semantic_search,
)
from agents.stt import transcribe_audio

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

groq_model = "openai/gpt-oss-120b"

CHAPTER_NUM_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)
ALL_VERSES_RE = re.compile(
    r"(?:\ball\b|\bevery\b|\bentire\b|\bfull\b|\bwhole\b)\s+(?:the\s+)?verses?"
    r"|verses?\s+of\s+(?:the\s+)?(?:whole|entire|full)\s+chapter"
    r"|chapter\s+all\s+verses?",
    re.IGNORECASE,
)

system_prompt = (
    "You are a Bhagavad Gita assistant. "
    "When the user asks to recite, get, teach, or practice a specific chapter and verse, "
    "call get_gita_verse with the chapter and verse numbers. "
    "When they ask for ALL verses in a chapter (e.g. 'recite chapter 12 all verses'), "
    "call recite_gita_chapter once with that chapter number — do not stop after one verse. "
    "When they quote shlok text or search by theme without chapter/verse, call search_gita. "
    "Present tool results faithfully: only Devanagari and English translation — "
    "do not invent IAST, synonyms, or purport. "
    "Keep replies concise. Mention the record button for live shlok practice."
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_gita_verse",
            "description": (
                "Fetch and recite a specific Bhagavad Gita verse. Use when the user asks to "
                "recite, get, teach, or practice a shlok with chapter and verse (e.g. chapter 1 verse 1)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter": {
                        "type": "integer",
                        "description": "Gita chapter number (1–18)",
                    },
                    "verse": {
                        "type": "integer",
                        "description": "Verse number within that chapter",
                    },
                },
                "required": ["chapter", "verse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_gita",
            "description": (
                "Semantic search the Gita when the user provides text or a theme "
                "without chapter and verse numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search text: quoted shlok, keywords, or theme",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recite_gita_chapter",
            "description": (
                "Recite every verse in a chapter in order. Use when the user asks for "
                "all verses, entire chapter, or full chapter recitation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter": {
                        "type": "integer",
                        "description": "Gita chapter number (1–18)",
                    },
                },
                "required": ["chapter"],
            },
        },
    },
]


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts).strip()
    return str(content)


def history_to_messages(history) -> list[dict]:
    """Convert Gradio chat history into OpenAI message list."""
    messages = []
    if not history:
        return messages
    for item in history:
        if isinstance(item, dict) and "role" in item:
            messages.append(
                {"role": item["role"], "content": _content_to_text(item.get("content"))}
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_text, bot_text = item
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if bot_text:
                messages.append({"role": "assistant", "content": str(bot_text)})
    return messages


RECORD_HINT = (
    "\n\nYou can press the **record** button to practice reciting this shloka aloud."
)


def _wants_full_chapter(message: str) -> int | None:
    """Detect 'recite chapter N all verses' without relying on the LLM."""
    m = CHAPTER_NUM_RE.search(message)
    if not m or not ALL_VERSES_RE.search(message):
        return None
    return int(m.group(1))


def _fetch_chapter_recitation(chapter: int, translation_on: bool) -> tuple[str, list[dict]]:
    verses = get_chapter_verses(collection, chapter)
    if not verses:
        return f"No verses indexed for Chapter {chapter}.", []
    body = format_search_results(verses, show_translation=translation_on)
    return body + RECORD_HINT, verses


def _format_recite_reply(verses: list[dict], translation_on: bool) -> str:
    if len(verses) == 1:
        body = format_verse_markdown(verses[0], show_translation=translation_on)
    else:
        body = format_search_results(verses, show_translation=translation_on)
    return body + RECORD_HINT


def handle_tool_calls(message, translation_on: bool) -> tuple[list[dict], list[dict]]:
    """Run tools; return OpenAI tool messages and verses queued for audio (no TTS here)."""
    responses = []
    verses_for_audio = []
    for tool_call in message.tool_calls or []:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")
        content = ""

        if name == "get_gita_verse":
            chapter = int(args.get("chapter", 0))
            verse = int(args.get("verse", 0))
            data = get_verse(collection, chapter, verse)
            if data:
                verses_for_audio.append(data)
                content = format_verse_markdown(data, show_translation=translation_on)
            else:
                content = f"No verse found for Chapter {chapter}, Verse {verse}."

        elif name == "search_gita":
            query = args.get("query", "")
            matches = semantic_search(collection, query)
            content = format_search_results(matches, show_translation=translation_on)

        elif name == "recite_gita_chapter":
            chapter = int(args.get("chapter", 0))
            content, batch = _fetch_chapter_recitation(chapter, translation_on)
            verses_for_audio.extend(batch)

        responses.append(
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call.id,
            }
        )
    return responses, verses_for_audio


def analyse_chat(
    message: str, history=None, translation_on: bool = True
) -> tuple[str, list[dict]]:
    """Chat with tool use. Returns (reply text, verses to recite after UI updates)."""
    history = history or []

    chapter = _wants_full_chapter(message)
    if chapter is not None:
        return _fetch_chapter_recitation(chapter, translation_on)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": message})

    response = groq.chat.completions.create(
        model=groq_model,
        messages=messages,
        tools=tools,
    )

    while response.choices[0].finish_reason == "tool_calls":
        msg = response.choices[0].message
        tool_responses, verses_for_audio = handle_tool_calls(msg, translation_on)
        messages.append(msg)
        messages.extend(tool_responses)

        tool_names = {tc.function.name for tc in (msg.tool_calls or [])}
        if verses_for_audio and tool_names <= {"get_gita_verse", "recite_gita_chapter"}:
            if "recite_gita_chapter" in tool_names:
                body = format_search_results(verses_for_audio, show_translation=translation_on)
                return body + RECORD_HINT, verses_for_audio
            return _format_recite_reply(verses_for_audio, translation_on), verses_for_audio

        response = groq.chat.completions.create(
            model=groq_model,
            messages=messages,
            tools=tools,
        )

    return response.choices[0].message.content or "", []


def play_recitation(verses: list[dict], translation_on: bool) -> None:
    """Play TTS after the chat UI has shown the verse text."""
    path = None
    for data in verses or []:
        path = recite_verse(data, with_translation=translation_on)
        if path:
            return path
    return None


def chat_with_audio(
    message: str,
    history,
    translation_on: bool,
    audio_path: str | None,
) -> tuple[list, str, list[dict]]:
    """Update chat first; verses for audio are played in a follow-up Gradio step."""
    if audio_path:
        spoken = transcribe_audio(audio_path)
        if spoken:
            message = f"{message}\n{spoken}".strip() if message else spoken
    if not message:
        return history, "", []
    reply, verses_for_audio = analyse_chat(message, history, translation_on)
    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, "", verses_for_audio
