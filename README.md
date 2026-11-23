# DocLens — PDF RAG Assistant

A small, local question-answering application for text-based PDFs. It uses Flask, plain HTML/CSS/JavaScript, OpenAI embeddings and chat, NumPy cosine similarity, and SQLite. No frontend framework or separate vector database is required.

## Run on Windows

Open PowerShell in this folder. Python 3.12 is recommended.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Replace `replace\_with\_your\_key` with your actual API key, keeping `OPENAI\_API\_KEY=` at the start of the line. Save and close Notepad. If you already have a working `.env`, copy it from the old project instead of overwriting it. Never commit `.env` or share screenshots of its contents. Revoke any key previously exposed. API usage requires available billing/credits.

```powershell
python app.py
```

Open http://127.0.0.1:5000. Stop any old Flask server first with Ctrl+C. Upload `sample.pdf`, ask **What does a solar inverter do?**, then ask **What is the author's passport number?**. The second question should be refused. Old indexed PDFs are not automatically migrated; upload them again. Only use PDFs you have permission to send to the API.

## Files

|File|Purpose|
|-|-|
|`app.py`|Flask routes, input validation, API configuration|
|`rag.py`|PDF extraction, chunking, embeddings, SQLite, search and answer generation|
|`index.html`|Complete responsive frontend (HTML, CSS and JavaScript in one file)|
|`evaluate.py`|Offline unit tests and optional live evaluation|
|`sample.pdf`|Two-page educational example|
|`requirements.txt`|Python dependencies|
|`.env.example`|API configuration template|
|`.gitignore`|Keeps secrets, local index and test results out of Git|
|`.github/workflows/tests.yml`|Runs offline tests on pushes and pull requests|

`storage/library.sqlite3` is created automatically. It stores extracted text and embeddings, not the original PDF. The UI can remove documents. Changing the embedding model requires re-indexing documents. This is a local single-user prototype, not an authenticated production service. Scanned PDFs need OCR; encrypted PDFs are rejected. No PDF or API credentials are included other than the public educational sample and placeholders.

## Technical explanation (English)

**RAG vs fine-tuning:** RAG retrieves relevant external information at inference time, without updating the language model's weights. Fine-tuning updates model parameters to change behavior using training examples. I chose RAG because PDFs can change frequently and answers need traceable source evidence.

**Pipeline:** PDF upload → pypdf text extraction → overlapping chunks of up to 900 characters → OpenAI `text-embedding-3-small` → SQLite → question embedding → cosine similarity → top four chunks → `gpt-4o-mini` → answer and cited excerpts. Chunk overlap reduces loss of context across boundaries; character-based chunking is easy to understand but can split words. Embeddings are numerical vectors representing semantic content.

**Cosine similarity:** `cos(q,v) = dot(q,v) / (norm(q) \* norm(v))`. Higher scores indicate closer vector directions, not a probability or answer accuracy. NumPy ranks the vectors. A linear search is simple and suitable for a small library; a dedicated vector index is a possible future improvement.

**Backend:** Flask serves the HTML and exposes `GET /api/documents`, `POST /api/upload`, `POST /api/ask`, and `POST /api/delete`. The frontend uses Fetch. The API key remains server-side. SQLite persists text and vectors between restarts. No external database setup is needed.

**Hallucination handling:** The model is instructed to use only retrieved excerpts, treat document instructions as untrusted, and abstain if evidence is insufficient. JSON output includes a `found` flag and source IDs. The backend validates source IDs; the UI distinguishes cited chunks from other retrieved chunks. Valid IDs do not guarantee factual correctness. Users should inspect supporting excerpts. The refusal is not a guarantee against all hallucinations or prompt injection.

## Evaluation

```powershell
python evaluate.py
python evaluate.py --live
```

The first command runs offline tests with a fake client, no key or paid API usage. The second uses OpenAI and the included sample PDF in a temporary database, leaving your own library unchanged. Six questions cover four answerable and two unanswerable cases. Results are printed and saved in `evaluation\_results.json`.

* Hit@4: fraction of answerable questions where a chunk containing the expected evidence is retrieved.
* Keyword pass rate: fraction of answerable questions with expected English terms and a positive response. A proxy, not verified factual accuracy.
* Evidence citation rate: fraction of answerable questions citing an expected evidence chunk.
* Abstention rate: fraction of unanswerable questions correctly refused.

The two-page sample is small and has few chunks, so Hit@4 may be trivial. For a meaningful evaluation, create a larger set of varied documents and questions, mark relevant excerpts, and manually label every answer as correct, partially correct, incorrect, or properly refused. Report answer accuracy (correct / answerable questions) and unsupported-answer rate (answers with unsupported claims / answers given). Do not invent scores. Include limitations and real outputs in your report.

## GitHub workflow

Create a new empty repository at https://github.com/new (no starter README). In PowerShell:

```powershell
git init
git add .
git status --short
git commit -m "Build PDF RAG assistant"
git branch -M main
git remote add origin https://github.com/YOUR\_USERNAME/YOUR\_REPO.git
git push -u origin main
```

Check `git status --short` before committing: `.env` and `storage/` must not be listed. If a secret was committed before, revoke it and remove it from Git history; ignoring it now is not sufficient. If Git requests identity, set your real name and email using `git config user.name` and `git config user.email`.

Practice a real feature branch: create it, make a small meaningful improvement (for example add a new evaluation question and expected evidence), run the tests, then commit it.

```powershell
git switch -c feature/add-evaluation-case
python evaluate.py
git add evaluate.py
git commit -m "Add a new evaluation case"
git push -u origin feature/add-evaluation-case
```

On GitHub, choose Compare \& pull request, review the changes and merge after checks pass. Then run `git switch main` and `git pull`. The workflow runs offline without API credentials. These instructions are not evidence of an actual repository or completed pull request; create and review them yourself.

## Short presentation script

My project is DocLens, a PDF question-answering system based on RAG. I chose Flask and vanilla JavaScript to keep the application easy to understand. I extract and split PDF text, create embeddings, and store them locally in SQLite. I use cosine similarity to retrieve the four most relevant chunks, then send them to a language model to generate an evidence-based answer. I display the supporting excerpts and page numbers so users can verify the output. My evaluation includes retrieval checks, answer keyword checks, citation checks, and missing-information tests. These automatic metrics have limitations, so manual verification is important. I use Git branches, commits and pull requests to track and review changes.

## Evaluation Results



The system was tested locally using six unit tests

and six live evaluation questions with the OpenAI API.



| Metric | Result |

|---|---|

| Unit Tests | 6/6 Passed |

| Live Evaluation | 6/6 Passed |

| Hit@4 | 100% |

| Keyword Pass Rate | 100% |

| Evidence Citation Rate | 100% |

| Abstention Rate | 100% |



The application was also tested with multiple PDF

documents, document filtering, source citations,

and questions with missing information.



These results are based on a small evaluation dataset

and do not guarantee performance on all documents.

