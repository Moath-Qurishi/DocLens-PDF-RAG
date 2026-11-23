import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from dotenv import load_dotenv
from rag import RAG, chunk_text, cosine_scores, NOT_FOUND

BASE = Path(__file__).resolve().parent
QUESTIONS = [
    ("What does a solar inverter do?", 2, "inverter converts", ["direct current", "alternating current"]),
    ("What do solar panels convert light into?", 1, "convert light into", ["direct current"]),
    ("What can batteries store?", 2, "batteries can store", ["electricity"]),
    ("Does solar output depend on weather?", 1, "depends on sunlight and weather", ["weather"]),
    ("What is the author's passport number?", None, None, []),
    ("What was the exact cost of installing the system?", None, None, []),
]


class FakeClient:
    def __init__(self):
        self.embeddings = NS(create=self.embed)
        self.chat = NS(completions=NS(create=self.complete))
        self.reply = dict(found=True, answer="An inverter converts DC to AC.", citations=[1])

    def embed(self, model, input):
        data = []
        for i, text in enumerate(input):
            vector = [float("inverter converts" in text.lower() or "inverter do" in text.lower()), 1.0]
            data.append(NS(index=i, embedding=vector))
        return NS(data=data)

    def complete(self, **kwargs):
        return NS(choices=[NS(message=NS(content=json.dumps(self.reply)))])


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = FakeClient()
        self.rag = RAG(Path(self.temp.name) / "test.db", self.client)

    def test_chunking(self):
        self.assertEqual(chunk_text("  "), [])
        text = "word " * 500
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 900 for c in chunks))

    def test_cosine(self):
        np.testing.assert_allclose(cosine_scores([1, 0], [[1, 0], [0, 1], [-1, 0], [0, 0]]), [1, 0, -1, 0])

    def test_index_and_replace(self):
        path = BASE / "sample.pdf"
        first = self.rag.index(path, "sample.pdf")
        self.assertEqual(first["pages"], 2)
        self.rag.index(path, "sample.pdf")
        self.assertEqual(self.rag.documents(), [first])
        self.rag.delete("sample.pdf")
        self.assertEqual(self.rag.documents(), [])

    def test_search_and_filter(self):
        self.rag.index(BASE / "sample.pdf", "sample.pdf")
        self.rag.index(BASE / "sample.pdf", "other.pdf")
        hits = self.rag.retrieve(QUESTIONS[0][0], "sample.pdf")
        self.assertEqual(hits[0]["page"], 2)
        self.assertTrue(all(h["document"] == "sample.pdf" for h in hits))
        self.assertEqual(self.rag.retrieve("question", "missing.pdf"), [])

    def test_answer_and_citation_validation(self):
        self.rag.index(BASE / "sample.pdf", "sample.pdf")
        result = self.rag.answer(QUESTIONS[0][0])
        self.assertTrue(result["found"])
        self.assertEqual(result["citations"], [1])
        self.client.reply["citations"] = [99]
        self.assertEqual(self.rag.answer("question")["answer"], NOT_FOUND)
        self.client.reply = dict(found=False, answer="Unknown", citations=[])
        self.assertFalse(self.rag.answer("missing information")["found"])

    def test_empty_library(self):
        self.assertFalse(self.rag.answer("question")["found"])


def live_evaluation():
    from openai import OpenAI
    load_dotenv(BASE / ".env")
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "replace_with_your_key":
        raise SystemExit("Set OPENAI_API_KEY in .env first.")
    results = []
    with tempfile.TemporaryDirectory() as folder:
        rag = RAG(Path(folder) / "evaluation.db", OpenAI(timeout=60, max_retries=1))
        rag.index(BASE / "sample.pdf", "sample.pdf")
        for question, page, phrase, keywords in QUESTIONS:
            response = rag.answer(question)
            relevant = [s for s in response["sources"] if page is not None and s["page"] == page and phrase in s["text"].lower()]
            cited = any(s["id"] in response["citations"] for s in relevant)
            passed = response["found"] and all(k in response["answer"].lower() for k in keywords) if page else not response["found"]
            results.append(dict(question=question, expected_page=page, expected_phrase=phrase,
                                retrieval_hit=bool(relevant) if page else None,
                                keyword_or_abstention_pass=passed, evidence_cited=cited if page else None,
                                answer=response["answer"], found=response["found"], citations=response["citations"], sources=response["sources"]))
            print(f"{'PASS' if passed else 'CHECK'} | {question}\n  {response['answer']}\n")
    positive = [r for r in results if r["expected_page"] is not None]
    negative = [r for r in results if r["expected_page"] is None]
    summary = dict(hit_at_4=sum(r["retrieval_hit"] for r in positive)/len(positive),
                   keyword_pass_rate=sum(r["keyword_or_abstention_pass"] for r in positive)/len(positive),
                   evidence_citation_rate=sum(r["evidence_cited"] for r in positive)/len(positive),
                   abstention_rate=sum(not r["found"] for r in negative)/len(negative))
    print(json.dumps(summary, indent=2))
    print("Keyword and citation checks are proxies; manually verify each factual claim. The tiny sample is not a benchmark.")
    (BASE / "evaluation_results.json").write_text(json.dumps(dict(summary=summary, results=results), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run six questions using the OpenAI API (paid usage).")
    args = parser.parse_args()
    if args.live:
        live_evaluation()
    else:
        unittest.main(argv=[__file__], verbosity=2)
