from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import hashlib
import json
import math
import random
import re
import uuid
import numpy as np
import pandas as pd
from .engine import METHODS, tokens

METRICS = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "answer_f1", "exact_match",
           "citation_validity", "citation_recall", "total_ms", "retrieval_ms", "generation_ms"]


def normalize_answer(answer):
    answer = re.sub(r"\[[^\]]+\]", "", answer)
    return " ".join(tokens(answer))


def score_result(result, question):
    expected = set(question["relevant_ids"])
    retrieved = [s["id"] for s in result["sources"]]
    relevance = [int(i in expected) for i in retrieved]
    hits = len(set(retrieved) & expected)
    recall = hits / len(expected) if expected else None
    precision = hits / len(retrieved) if expected and retrieved else (0.0 if expected else None)
    mrr = next((1 / (i + 1) for i, value in enumerate(relevance) if value), 0.0) if expected else None
    dcg = sum(v / math.log2(i + 2) for i, v in enumerate(relevance))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), len(retrieved))))
    ndcg = dcg / ideal if ideal else (0.0 if expected else None)
    prediction = normalize_answer(result["answer"])
    reference = normalize_answer(question["reference_answer"])
    common = sum((Counter(prediction.split()) & Counter(reference.split())).values())
    f1 = 2 * common / (len(prediction.split()) + len(reference.split())) if prediction or reference else 1.0
    citations = re.findall(r"\[([^\]]+)\]", result["answer"])
    validity = sum(c in retrieved for c in citations) / len(citations) if citations else 0.0
    citation_recall = len(set(citations) & expected) / len(expected) if expected else None
    return {"recall_at_k": recall, "precision_at_k": precision, "mrr": mrr, "ndcg_at_k": ndcg,
            "answer_f1": f1, "exact_match": float(prediction == reference),
            "citation_validity": validity, "citation_recall": citation_recall}


def benchmark(engine, questions, progress=lambda _: None, cancelled=lambda: False, methods=None):
    rng = random.Random(engine.settings.seed)
    rows = []
    for idx, question in enumerate(questions):
        if cancelled():
            raise InterruptedError("Benchmark cancelled; no incomplete run was saved.")
        order = list(methods or METHODS)
        rng.shuffle(order)
        progress(f"Question {idx + 1}/{len(questions)}")
        for result in engine.compare(question["question"], progress, cancelled, order):
            rows.append(result | score_result(result, question) | {"question_id": question["id"],
                         "reference_answer": question["reference_answer"],
                         "relevant_ids": question["relevant_ids"], "origin": question.get("origin", "unspecified")})
    frame = pd.DataFrame(rows)
    summary = []
    for method in (methods or METHODS):
        group = frame[frame.method == method]
        entry = {"method": method, "questions": len(group)}
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            entry[metric] = float(np.mean(values)) if len(values) else None
            entry[metric + "_n"] = len(values)
            # Question-level bootstrap describes variability within this supplied benchmark.
            if len(values) > 1:
                boot_rng = np.random.default_rng(engine.settings.seed)
                means = np.mean(boot_rng.choice(values, (1000, len(values)), replace=True), axis=1)
                entry[metric + "_ci95"] = np.quantile(means, [0.025, 0.975]).tolist()
            else:
                entry[metric + "_ci95"] = None
        summary.append(entry)
    return rows, summary


def save_run(directory, engine, corpus_hash, questions, rows, summary):
    from datetime import timedelta
    timestamp = datetime.now(timezone.utc)
    local = timestamp.astimezone(timezone(timedelta(hours=8)))
    def slug(text):
        return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:70]
    rag = summary[0]["method"] if len(summary) == 1 else "all-rags"
    llm = engine.settings.answer_model if engine.settings.mode == "ollama" else "offline-demo"
    run_id = local.strftime("%Y%m%d_%H%M%S_%f") + "_" + slug(llm) + "_" + slug(rag)
    folder = Path(directory) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "PyQt6", "requests")}
    run = {"run_id": run_id, "created_utc": timestamp.isoformat(), "config": engine.config(),
           "corpus_sha256": corpus_hash, "benchmark_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest(),
           "package_versions": versions, "questions": questions, "results": rows, "summary": summary,
           "notes": ["Starter questions are derived from indexed metadata; they are a workflow check, not an independent benchmark.",
                     "Question embedding is computed once and its measured time is charged equally to each method; sums are not wall-clock duration.",
                     "Benchmarks generate answers independently per method; interactive identical-context answer reuse is disabled.",
                     "Citation validity checks ID membership, not whether claims are supported.",
                     "Answer F1 measures token overlap, not semantic truth or safety.",
                     "Retrieval timing includes query embedding and reranking; index construction is reported separately.",
                     "Seeded method order is shuffled per question; latency includes cold model requests.",
                     "95% bootstrap intervals reflect this benchmark's question sampling only."]}
    (folder / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    flat_rows = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows]
    pd.DataFrame(flat_rows).to_csv(folder / "results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
    (folder / "validation.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([q | {"relevant_ids": json.dumps(q["relevant_ids"])} for q in questions]).to_csv(folder / "validation.csv", index=False, encoding="utf-8-sig")
    (folder / "corpus.json").write_text(json.dumps([{"id": d.id, "title": d.title, "text": d.text, "source": d.source} for d in engine.docs], indent=2, ensure_ascii=False), encoding="utf-8")
    return folder, run
