import modal 

app = modal.App("gita")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libssl-dev","libasound2","wget")
    .pip_install("chromadb","gradio","gTTS","indic_transliteration","kokoro_onnx","numpy","openai","pygame","pypdf","python-dotenv","scipy","sentence_transformers","sounddevice","transformers","torch","peft","bitsandbytes","accelerate","azure-cognitiveservices-speech")
    .add_local_dir(".","/app", ignore=[".*","*.pyc","__pycache__","*.log"])
)
secrets = [
    modal.Secret.from_name("AZURE_SECRET"),
    modal.Secret.from_name("AZURE_REGION"),
    modal.Secret.from_name("HF_TOKEN"),
    modal.Secret.from_name("GROQ_API_KEY"),
]
@app.function(image=image,secrets=secrets,max_containers=1)

@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def ui_modal():
    import sys
    sys.path.insert(0,"/app")

    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from gita import ui

    return mount_gradio_app(app=FastAPI(),blocks=ui, path="/")

