from fastapi import FastAPI, UploadFile, File
from pdf_loader import load_and_chunk
from rag import ingest_documents, answer_question
import shutil
import os

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_and_chunk(path)
    ingest_documents(chunks)

    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/ask")
def ask(question: str):
    answer = answer_question(question)
    return {"answer": answer}
