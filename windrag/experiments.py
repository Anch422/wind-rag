"""Sequential experiments, validation sampling, run discovery, and graph exports."""
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import random
import re
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from .engine import Engine, METHODS
from .evaluation import benchmark, save_run, METRICS
from .data import starter_questions

LABELS = {"recall_at_k": "Recall@k", "precision_at_k": "Precision@k", "mrr": "MRR", "ndcg_at_k": "nDCG@k",
          "answer_f1": "Answer token F1", "exact_match": "Exact match", "citation_validity": "Citation ID validity",
          "citation_recall": "Citation recall", "total_ms": "Total latency (ms)",
          "retrieval_ms": "Retrieval latency (ms)", "generation_ms": "Generation latency (ms)"}


def metric_label(metric):
    return LABELS.get(metric, metric) + (" · lower is better" if metric.endswith("_ms") else " · higher is better")


def validation_set(events, features, size=24, seed=42, report_docs=None):
    rows = starter_questions(events, features, report_docs)
    pools = defaultdict(list)
    for row in rows:
        kind = 'report' if row['relevant_ids'][0].startswith('REPORT:') else 'sensor' if ':sensor:' in row['id'] else 'event'
        pools[(row.get('farm', row['relevant_ids'][0][0]), kind)].append(row)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while any(pools.values()) and len(selected) < size:
        for key in sorted(pools):
            if pools[key] and len(selected) < size:
                row = pools[key].pop()
                row["origin"] = "generated validation: balanced farm/type sample; requires human review"
                selected.append(row)
    return selected


def plot_metric(entries, metric, figure=None):
    figure = figure or Figure(figsize=(10, 5), tight_layout=True)
    figure.clear()
    ax = figure.add_subplot(111)
    values = [entry.get(metric) for entry in entries]
    colors = ["#2563eb", "#0d9488", "#f59e0b", "#8b5cf6"]
    bars = ax.bar(range(len(entries)), [0 if v is None else v for v in values],
                  color=[colors[i % len(colors)] for i in range(len(entries))])
    import textwrap
    labels = ["\n".join(textwrap.wrap(e.get("label", e.get("method", "")), width=25)) for e in entries]
    ax.set_xticks(range(len(entries)), labels, fontsize=8)
    ax.set_title(metric_label(metric))
    ax.set_ylabel("Milliseconds" if metric.endswith("_ms") else "Score (0–1)")
    if not metric.endswith("_ms"):
        ax.set_ylim(0, 1.15)
    else:
        ax.set_ylim(0, max([v or 0 for v in values] + [1]) * 1.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), "N/A" if value is None else f"{value:.3f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.15)
    return figure


def export_graphs(folder, entries, rows=None):
    folder = Path(folder)
    graphs = folder / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(entries, metric).savefig(graphs / f"{metric}.png", dpi=160)
    fig = Figure(figsize=(9, 5), tight_layout=True)
    ax = fig.add_subplot(111)
    for entry in entries:
        x, y = entry.get("total_ms"), entry.get("answer_f1")
        if x is not None and y is not None:
            ax.scatter(x, y, s=70)
            ax.annotate(entry.get("label", entry.get("method", "")), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total latency (ms) · lower is better ←")
    ax.set_ylabel("Answer token F1 · higher is better ↑")
    ax.set_title("Answer overlap versus response time")
    ax.grid(alpha=0.2)
    fig.savefig(graphs / "quality_vs_latency.png", dpi=160)
    if rows:
        per_question = Figure(figsize=(12, 7), tight_layout=True)
        quality = per_question.add_subplot(211)
        latency = per_question.add_subplot(212)
        frame = pd.DataFrame(rows)
        group_fields = ["model", "method"] if "model" in frame.columns else ["method"]
        for label, group in frame.groupby(group_fields, sort=False):
            question_values = group.groupby("question_id", sort=False)[["answer_f1", "total_ms"]].mean()
            name = " / ".join(label) if isinstance(label, tuple) else str(label)
            quality.plot(range(1, len(question_values) + 1), question_values.answer_f1, marker="o", markersize=3, label=name)
            latency.plot(range(1, len(question_values) + 1), question_values.total_ms, marker="o", markersize=3, label=name)
        quality.set_title("Per-question answer token F1 · higher is better")
        quality.set_ylim(0, 1.1)
        latency.set_title("Per-question latency (ms) · lower is better")
        for axis in (quality, latency):
            axis.set_xlabel("Validation question number (source CSV supplies IDs)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        per_question.savefig(graphs / "per_question_metrics.png", dpi=160)


def sequential_runs(engine, corpus_hash, questions, models, methods, directory,
                    progress=lambda _: None, cancelled=lambda: False, completed=lambda _: None, staging=None):
    if not models or not methods or not questions:
        raise ValueError("Choose models, RAG algorithms, and validation questions first.")
    folders = []
    total = len(models) * len(methods)
    for model in models:
        for rag in methods:
            if cancelled():
                raise InterruptedError("Testing cancelled. Previously completed runs remain saved.")
            settings = replace(engine.settings, answer_model=model)
            engine.apply_settings(settings)
            prefix = f"Run {len(folders) + 1}/{total} · {model} · {rag}"
            try:
                rows, summary = benchmark(engine, questions, lambda msg: progress(prefix + " · " + msg), cancelled, methods=[rag])
            except InterruptedError:
                raise
            except Exception as exc:
                stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
                slug = lambda value: re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")[:70]
                folder = Path(directory) / f"{stamp}_{slug(model)}_{slug(rag)}_failed"
                folder.mkdir(parents=True, exist_ok=False)
                failure = {"run_id": folder.name, "status": "failed", "config": engine.config(),
                           "algorithm": rag, "error": str(exc), "questions": questions,
                           "results": [], "summary": []}
                (folder / "run.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
                (folder / "validation.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
                (folder / "report.md").write_text(f"# Failed run\n\n{model} / {rag}\n\n{exc}\n\nNo successful metrics recorded.\n", encoding="utf-8")
                folders.append(str(folder))
                completed(str(folder))
                progress(f"{prefix} failed; error saved. Continuing with the next selection.")
                continue
            if cancelled():
                raise InterruptedError("Testing cancelled. Previously completed runs remain saved.")
            folder, run = save_run(staging if staging is not None else directory, engine, corpus_hash, questions, rows, summary)
            export_graphs(folder, summary, rows)
            (folder / "report.md").write_text(f"# {model} / {rag}\n\n{len(questions)} validation questions.\n\n" +
                "\n".join(f"- {metric_label(m)}: {summary[0].get(m)}" for m in METRICS) +
                "\n\nGenerated validation requires independent human review. Citation validity checks IDs, not factual support.\n", encoding="utf-8")
            if staging is not None:
                if cancelled():
                    raise InterruptedError('Benchmark cancelled. Completed runs remain saved.')
                destination = Path(directory) / folder.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                folder.rename(destination)
                folder = destination
            folders.append(str(folder))
            completed(str(folder))
    return folders


def scan_runs(directory):
    entries, errors = [], []
    for path in sorted(Path(directory).rglob("run.json"), reverse=True):
        if any(part in ('comparisons', '.in_progress') for part in path.relative_to(directory).parts):
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
            if not run.get("results") or not run.get("summary"):
                raise ValueError("No complete results")
            for summary in run["summary"]:
                model = run["config"]["answer_model"] if run["config"]["mode"] == "ollama" else "offline-demo"
                entries.append({"path": str(path), "run": run, "summary": summary,
                                "model": model, "rag": summary["method"],
                                "label": f"{model} / {summary['method']}", "run_id": run["run_id"]})
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path.parent.name}: {exc}")
    entries.sort(key=lambda entry: entry["run"].get("created_utc", ""), reverse=True)
    return entries, errors


def aggregate(entries):
    if not entries:
        raise ValueError("Select at least one saved run.")
    identities = {(e["run"].get("corpus_sha256"), e["run"].get("benchmark_sha256"),
                   e["run"]["config"].get("mode"), e["run"]["config"].get("embedding_model"),
                   e["run"]["config"].get("top_k"), e["run"]["config"].get("candidates"),
                   e["run"]["config"].get("temperature"), e["run"]["config"].get("prompt_version")) for e in entries}
    if len(identities) != 1:
        raise ValueError("These runs used different validation data, corpus, or retrieval/generation settings. Select matching runs for a fair comparison.")
    groups = defaultdict(list)
    for entry in entries:
        rows = [row for row in entry["run"]["results"] if row["method"] == entry["rag"]]
        groups[(entry["model"], entry["rag"])].extend(rows)
    summaries, all_rows = [], []
    for (model, rag), rows in groups.items():
        summary = {"label": f"{model} / {rag}", "model": model, "method": rag,
                   "runs": sum(e["model"] == model and e["rag"] == rag for e in entries), "observations": len(rows)}
        frame = pd.DataFrame(rows)
        for metric in METRICS:
            # Average repeated runs per question first so repeated tests cannot overweight a question.
            values = frame.groupby("question_id")[metric].mean().dropna().to_numpy(dtype=float)
            summary[metric] = float(np.mean(values)) if len(values) else None
            summary[metric + "_n"] = len(values)
        summaries.append(summary)
        all_rows.extend(row | {"model": model, "rag": rag} for row in rows)
    return summaries, all_rows


def export_comparison(directory, entries, name="comparison"):
    summaries, rows = aggregate(entries)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:60] or "comparison"
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    folder = Path(directory) / "comparisons" / f"{stamp}_{safe}"
    folder.mkdir(parents=True, exist_ok=False)
    data = {"source_runs": [{"path": e["path"], "run_id": e["run_id"], "rag": e["rag"]} for e in entries],
            "summary": summaries, "results": rows,
            "aggregation": "Repeated runs averaged per question, then across questions; descriptive comparison, not significance test."}
    (folder / "comparison.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (folder / "validation.json").write_text(json.dumps(entries[0]["run"]["questions"], indent=2, ensure_ascii=False), encoding="utf-8")
    snapshots = folder / "source_runs"
    snapshots.mkdir()
    seen = set()
    for entry in entries:
        if entry["path"] not in seen:
            seen.add(entry["path"])
            (snapshots / f"{len(seen):03d}.json").write_text(json.dumps(entry["run"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(folder / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows]).to_csv(folder / "results.csv", index=False, encoding="utf-8-sig")
    export_graphs(folder, summaries, rows)
    (folder / "report.md").write_text("# Run comparison\n\n" + data["aggregation"] + "\n\n" + "\n".join(f"- {e['run_id']} / {e['rag']}" for e in entries), encoding="utf-8")
    return folder
