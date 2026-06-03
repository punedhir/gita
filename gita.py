import gradio as gr

from rag import collection, format_verse_markdown, get_catalog, get_verse
from text import chat_with_audio, play_recitation

catalog = get_catalog(collection)
chapters = sorted(catalog.keys()) or [1]


def verses_for_chapter(chapter: int) -> list[int]:
    return catalog.get(int(chapter), [1])


def update_verse_dropdown(chapter: int):
    verses = verses_for_chapter(chapter)
    return gr.Dropdown(choices=verses, value=verses[0] if verses else 1)


def show_verse_panel(chapter: int, verse: int) -> str:
    """Browse panel always shows Devanagari + English translation."""
    data = get_verse(collection, int(chapter), int(verse))
    return format_verse_markdown(data, show_translation=True)


def practice_selected_shlok(chapter: int, verse: int) -> tuple[str, list]:
    """Show full text; play Sanskrit only on submit."""
    data = get_verse(collection, int(chapter), int(verse))
    if not data:
        return "_Verse not found._", []
    md = format_verse_markdown(data, show_translation=True)
    return md, [data]


def submit_chat(message, history, translation_on, audio_path):
    return chat_with_audio(message, history, translation_on, audio_path)


CUSTOM_CSS = """
/* Page */
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    background-color: #f3ebe0 !important;
    max-width: 100% !important;
}

/* Title banner — high contrast on its own dark strip */
.gita-header {
    text-align: center;
    padding: 1.1rem 1rem 1rem;
    margin-bottom: 0.85rem;
    border-radius: 10px;
    background: linear-gradient(135deg, #3d2518 0%, #5c3822 50%, #3d2518 100%);
    border: 2px solid #d4a017;
    box-shadow: 0 4px 14px rgba(45, 28, 16, 0.35);
}
.gita-header h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffecb3;
    text-shadow: 0 2px 4px #1a0f08, 0 0 24px rgba(255, 213, 79, 0.35);
    letter-spacing: 0.06em;
}
.gita-header p {
    margin: 0.4rem 0 0;
    font-size: 0.95rem;
    color: #fff8e7;
    font-weight: 500;
}

/* Main chat — sized to leave room for browse panel on one screen */
.gita-chat-panel {
    border: 1px solid #c4a574 !important;
    border-radius: 12px !important;
    background-color: #fffdf8 !important;
    padding: 0.75rem 1rem 1rem !important;
    box-shadow: 0 2px 12px rgba(61, 37, 24, 0.12);
}
.gita-chat-panel .bubble-wrap,
.gita-chat-panel .widget {
    max-height: min(38vh, 400px) !important;
}

/* Compact browse strip below chat */
.gita-browse-panel {
    border: 1px solid #b8956a !important;
    border-radius: 10px !important;
    background-color: #faf5ec !important;
    padding: 0.65rem 1rem 0.75rem !important;
    margin-top: 0.65rem !important;
    box-shadow: 0 2px 8px rgba(61, 37, 24, 0.1);
}
.gita-browse-panel h3, .gita-browse-panel .prose h3 {
    margin: 0 0 0.5rem 0 !important;
    font-size: 1rem !important;
    color: #3d2518 !important;
    font-weight: 600 !important;
}
.gita-browse-panel label, .gita-browse-panel span {
    color: #4a3220 !important;
}
.gita-verse-preview {
    max-height: none !important;
    overflow: visible !important;
    padding: 0.5rem 0.65rem;
    margin-top: 0.35rem;
    border-radius: 6px;
    background-color: #fffdf8;
    border: 1px solid #e0d0b8;
    font-size: 0.92rem;
    line-height: 1.5;
    color: #2c1810;
}
.gita-verse-preview .prose,
.gita-verse-preview .markdown,
.gita-verse-preview p,
.gita-verse-preview h3 {
    max-height: none !important;
    overflow: visible !important;
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    margin: 0.25rem 0 !important;
}
.gita-verse-preview h3 {
    font-size: 1rem !important;
    color: #3d2518 !important;
}
.gita-practice-btn {
    align-self: end;
    min-width: 7rem;
}

footer { display: none !important; }
"""

CUSTOM_JS = """
() => {
    document.querySelectorAll('.gita-chat-panel textarea, .gita-browse-panel input').forEach(el => {
        el.style.borderRadius = '8px';
    });
    document.querySelectorAll('.gita-verse-preview, .gita-verse-preview *').forEach(el => {
        el.style.maxHeight = 'none';
        el.style.overflow = 'visible';
        el.style.whiteSpace = 'normal';
        el.style.wordWrap = 'break-word';
    });
}
"""

with gr.Blocks(title="Shri Bhagavad Gita", fill_width=True) as ui:
    gr.HTML(
        """
        <div class="gita-header">
            <h1>श्रीमद् भगवद् गीता</h1>
            <p>Ask for a verse, search by meaning, or browse the catalog</p>
        </div>
        """
    )

    with gr.Column():
        with gr.Column(elem_classes=["gita-chat-panel"]):
            chatbot = gr.Chatbot(label="Gita Chat", height=380, scale=1)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="e.g. Recite chapter 1 verse 1, or search karma yoga…",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                translation_on = gr.Checkbox(
                    label="English translation ON",
                    value=True,
                )
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record / Upload",
                    scale=2,
                )
                audio_output = gr.Audio(label="Recitation",type="filepath", autoplay=True, interactive=False)

        with gr.Column(elem_classes=["gita-browse-panel"]):
            gr.Markdown("### Browse verses")
            with gr.Row(equal_height=True):
                chapter_dd = gr.Dropdown(
                    choices=chapters,
                    value=chapters[0],
                    label="Chapter",
                    scale=2,
                )
                verse_dd = gr.Dropdown(
                    choices=verses_for_chapter(chapters[0]),
                    value=verses_for_chapter(chapters[0])[0],
                    label="Verse",
                    scale=2,
                )
                practice_btn = gr.Button(
                    "Practice",
                    variant="secondary",
                    scale=1,
                    elem_classes=["gita-practice-btn"],
                )
            verse_md = gr.Markdown(
                value=show_verse_panel(
                    chapters[0], verses_for_chapter(chapters[0])[0]
                ),
                elem_classes=["gita-verse-preview"],
            )

    chapter_dd.change(
        update_verse_dropdown,
        inputs=[chapter_dd],
        outputs=[verse_dd],
    ).then(
        show_verse_panel,
        inputs=[chapter_dd, verse_dd],
        outputs=[verse_md],
    )
    verse_dd.change(
        show_verse_panel,
        inputs=[chapter_dd, verse_dd],
        outputs=[verse_md],
    )

    verses_pending = gr.State([])

    submit_inputs = [msg, chatbot, translation_on, audio_input]
    submit_outputs = [chatbot, msg, verses_pending]

    send_evt = send_btn.click(submit_chat, submit_inputs, submit_outputs)
    msg_evt = msg.submit(submit_chat, submit_inputs, submit_outputs)
    for evt in (send_evt, msg_evt):
        evt.then(
            lambda v, t: play_recitation(v, t),
            inputs=[verses_pending, translation_on],
            outputs=[audio_output],
        )

    practice_evt = practice_btn.click(
        practice_selected_shlok,
        inputs=[chapter_dd, verse_dd],
        outputs=[verse_md, verses_pending],
    )
    practice_evt.then(
        lambda verses: play_recitation(verses,False),
        inputs=[verses_pending],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    ui.launch(
        inbrowser=True,
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="orange",
            neutral_hue="stone",
        ),
    )
