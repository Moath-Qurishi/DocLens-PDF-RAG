import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI, OpenAIError
from pypdf.errors import PdfReadError

from rag import RAG

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 11 * 1024 * 1024
rag = RAG()


def api_client():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "replace_with_your_key":
        raise ValueError("Set OPENAI_API_KEY in your .env file and restart the app.")
    rag.client = OpenAI(api_key=key, timeout=60, max_retries=1)


@app.get("/")
def home():
    return send_from_directory(BASE, "index.html")


@app.get("/api/documents")
def documents():
    return jsonify(rag.documents())


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify(error="Choose a PDF file."), 400
    if file.stream.read(5) != b"%PDF-":
        return jsonify(error="Invalid PDF file."), 400
    file.stream.seek(0, 2)
    if file.stream.tell() > 10 * 1024 * 1024:
        return jsonify(error="Maximum PDF size is 10 MB."), 400
    file.stream.seek(0)
    name = file.filename.replace("\\", "/").split("/")[-1]
    try:
        api_client()
        return jsonify(rag.index(file.stream, name))
    except (ValueError, PdfReadError) as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Invalid request."), 400
    question = data.get("question", "")
    document = data.get("document")
    if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
        return jsonify(error="Enter a question between 1 and 2000 characters."), 400
    if document is not None and not isinstance(document, str):
        return jsonify(error="Invalid document."), 400
    try:
        api_client()
        return jsonify(rag.answer(question.strip(), document))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/delete")
def delete():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("document"), str):
        return jsonify(error="Choose a document."), 400
    rag.delete(data["document"])
    return jsonify(ok=True)


@app.errorhandler(OpenAIError)
def openai_error(exc):
    app.logger.warning("OpenAI request failed: %s", type(exc).__name__)
    status = getattr(exc, "status_code", None)
    message = {401: "Invalid API key. Check .env and restart.",
               429: "OpenAI rate limit or quota reached. Check API billing and usage."}.get(
                   status, "OpenAI request failed. Check your network and model settings.")
    return jsonify(error=message), 502


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Maximum PDF size is 10 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
