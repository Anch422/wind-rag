# LOUIE — LLM-powered Operational Understanding and Inspection Expert

Chat with wind turbine records and technician PDF reports, benchmark selected local models and RAG algorithms sequentially, and compare saved experiments.

Read [the beginner course](docs/COURSE.md) for explanations of Python, Conda, libraries, RAG, embeddings, metrics, and research methodology. This README describes the current interface.

## Start

Open Anaconda Prompt:

```bat
conda activate wind-rag
cd /d C:\Users\USER\Development\_THESIS
python app.py
```

The environment is outside the project folder. Missing app libraries can be installed with `python -m pip install -r requirements.txt` after activation.

## 1. Setup

Choose the extracted `datasets/CARE_To_Compare` folder and an embedding model. Ollama models are discovered automatically; **Refresh models** updates the lists.

**Include technician PDF reports** is enabled by default, with the separate folder `datasets/technician_reports`. Use **Browse reports** to select another reports folder, or uncheck the option for CARE-only experiments. Report changes automatically load a matching index. The original CARE files stay unchanged.

Opening the app starts **Initializing LOUIE** automatically. The progress bar covers model discovery, dataset loading, saved-index loading or building, and scanning saved results. Wait for initialization to finish; the app then unlocks Chat and Benchmark. You do not need to click a preparation button. The first embedding build can take longer than loading a saved index.

![Automatic initialization](docs/app-initializing.png)

Setup has two sub-tabs: **General** for dataset/model settings and **RAG indexes** for readiness and index details. The table explains what each algorithm does and which index it needs. The 450 source documents cover 95 events and 355 sensor definitions. Smaller windows provide scrolling so controls remain readable. **Reset to defaults** is in General. **Refresh / retry indexes** is an optional manual refresh, useful after changing metadata files on disk.

The app prepares dense vectors, BM25 term statistics, word TF-IDF, and character TF-IDF together, and saves them in `.cache/indexes/`. All six algorithms share these compatible index components. Switching algorithms or answer models does **not** embed the documents again. MMR and reranking operate at question time and need no separate index. Saved indexes are loaded automatically on each restart. Enabling reports changes the corpus, so the first combined index needs building; later openings reuse it. The former CARE-only cache remains available.

Answer models, top-k, candidates, temperature, and run seed can change without rebuilding. Changed dataset metadata, embedding models, server, or run mode require a matching index. If one exists, it is loaded; otherwise the app builds and saves it. Changing the dataset folder, mode, or embedding setup schedules automatic initialization after a short pause; finish text-field edits by pressing Tab or clicking another control. Dataset contents are also checked before chat or benchmarking, so changed metadata cannot silently use an old index.

**Reset to defaults** restores the standard dataset path, Ollama mode, localhost server, nomic embedding model, top-k 3, candidate pool 10, seed 42, temperature 0, and validation size 24. It selects the installed qwen2.5:7b answer model when available, otherwise the first available model. The original three algorithms are checked by default; the added algorithms are optional. Saved indexes, results, and validation questions are kept. A matching index is prepared automatically if these settings changed.

If initialization fails, the error stays visible and Setup becomes available. Start Ollama or correct the model/path, then use **Retry initialization**. Alternatively, choose **Offline demo**; its index initializes automatically without Ollama. Errors never silently switch your research run to another mode.

For a first trial, choose **Offline demo**. It uses statistical retrieval and extractive replies, not a real language model. Celsius/degree symbols are repaired in the indexed metadata without changing original dataset files.

![Setup General](docs/app-setup-general.png)

![RAG indexes and readiness](docs/app-setup-indexes.png)

## 2. Chat

Choose **one model** and **one RAG algorithm**, type a question, and press **Enter** or click **Send message**. **Shift+Enter** adds a new line. An animated typing indicator appears while the reply loads. Your message appears on the right; the reply appears on the left. Only the selected algorithm runs. **Show evidence** expands the retrieved records.

In **Dataset**, click a column title to sort the records; click again to reverse the order. The **Results** tab lets you view saved runs and compare selected runs.

Try: `What does sensor_0 measure at Wind Farm A, and what is its unit?`

### Technician reports and repair questions

The report dataset contains **45 separate searchable PDFs, one report per file, 90 pages total**, covering every recorded CARE anomaly. Each report has an inspection page and a repair/verification page, with varied 2023-2024 service dates, six companies and twelve technicians. It records the affected component, inspection findings, a possible cause, a repair record, parts/materials, verification, and limitations. Filenames include service date, farm, event and report ID. `manifest.json` maps each file to its event and PDF pages.

Generation history and verification status are retained in `manifest.json`, outside the indexed PDF text. These generated research records are not independently verified CARE field-service evidence. Their creation does not establish what actually caused or repaired a CARE anomaly.

Ask, for example:

- `How should I repair generator bearing damage at Wind Farm A, according to the technician reports?`
- `What possible cause was documented in technician report TR-A-040?`
- `What repair was documented for a faulty pitch-motor fan at Wind Farm C?`

Repair, fix, replacement, maintenance, cause, and technician-report questions automatically search **report evidence only**. An explicit farm narrows that report search. Other questions use the combined corpus. Each selected algorithm ranks the same eligible report evidence using its own retrieval method. The reply shows **Technician PDF reports**, and **Show evidence** displays report ID, PDF page, chunk citation and source file. If matching reports are unavailable, the app reports insufficient evidence rather than inventing a repair from metadata.

Ollama is instructed to ground recommendations in the retrieved repair records, keep possible causes uncertain, and describe the documented action directly without unrelated dataset commentary. The prompt asks it to request the component/event/symptom if the repair question is too vague. Offline demo extracts the top report section; it does not perform generative reasoning. Evaluate responses yourself before drawing research conclusions.

![Report-grounded repair chat](docs/app-report-chat.png)

Under **Show evidence**, PDF sources include an **Open PDF** link. Click it to open the report in your default Windows PDF viewer. The evidence title shows the cited page; missing or moved files display a clear message.

To add your own technician reports, place searchable `.pdf` files in a separate folder and choose it in Setup. Subfolders are scanned recursively. Text is extracted with `pypdf`; long pages become overlapping 300-word chunks with 50-word overlap. Citations preserve file and page. Scanned PDFs without searchable text need OCR first; encrypted or unreadable PDFs produce a clear initialization error. Unchanged PDF text is cached in memory to avoid reparsing it for every chat request. Use **Refresh / retry indexes** after editing files on disk.

### Files excluded from GitHub

Datasets, generated benchmark questions/results, saved chat sessions, indexes, temporary files, environments, and credentials are excluded by `.gitignore`. A fresh clone therefore needs the CARE dataset extracted into `datasets/CARE_To_Compare` (or selected in Setup). Provide searchable technician PDFs separately, or disable **Include technician PDF reports**. To regenerate the provided research reports after supplying CARE, install `reportlab` and run `python tools/generate_technician_reports.py`. The source code, tests, documentation, and small guide screenshots are included.

### How questions select evidence

Questions such as **“What's the problem with Wind Farm C?”**, **“What is wrong with Farm C?”**, or **“List anomalies at Wind Farm C”** search recorded anomaly events rather than sensor descriptions. Normal-event requests search normal records; broad health/status questions search event records of both labels. Naming farms restricts retrieval to those farms, including comparisons such as “farms A and C”. Explicit event IDs and turbine/asset numbers further restrict the evidence. Repair, troubleshooting, and possible-cause questions search technician reports. Sensor and unit questions retain general retrieval.

Answers summarize the selected historical records; they do not establish current turbine status or guarantee a complete anomaly inventory. This shared question routing is applied equally to all six retrieval algorithms, so benchmarking still compares their ranking methods. Unrecognized wording uses general retrieval; routing is deterministic and cannot understand every possible phrasing. Include farm, event, component, and dates in your question when relevant. Dates currently help ranking rather than acting as strict date filters.

### Saved chats

The **Your chats** sidebar on the left works like a chat history:

1. Click **New chat** to start a separate conversation.
2. Send a question. The first question becomes the chat's title.
3. Click a chat in the sidebar to return to it.
4. Close and reopen LOUIE: your last selected chat opens automatically after initialization.
5. Click **Delete chat** to permanently remove the selected conversation. Other chats and benchmark results are kept. Deleting the last chat creates an empty chat.

Messages and replies save immediately. Retrieved evidence, PDF links, model/RAG selections, and unsent drafts are also saved automatically. Drafts save shortly after typing and when switching chats or closing the app. Chats are stored locally in `sessions/chat_history.sqlite3`; keep this file if you back up or move your project. They do not require an online account. PDF links refer to the original files, so those files must remain available to open them.

Each request answers the current question independently; include the sensor/farm/event explicitly in follow-up questions. Saved history lets you revisit conversations, but previous messages are not added to the model's question automatically. Chat is separate from benchmark experiments and does not automatically create a results folder.

![Chat](docs/app-chat.png)

## 3. Benchmark

1. Check which models to test.
2. Check which RAG algorithms to test.
3. Choose a validation size and click **Generate validation**, or import JSON/CSV questions.
4. Review questions, reference answers, and relevant source IDs. Double-click cells to edit.
5. Export validation if you want a standalone copy.
6. Click **Start sequential benchmark**.

Two models and three algorithms produce **six separate runs**, executed one at a time. Each uses the same validation set and index. Completed runs remain saved if a later run fails or testing is cancelled. No benchmark answer reuse occurs across algorithms.

**Force cancel benchmark** stops the isolated benchmark process immediately, including an active model request. It does not wait for the model's response or timeout. Fully exported runs remain saved; the current incomplete run is discarded. The app becomes available again after the process stops. Index preparation and chat retain their usual cancellation behavior.

If a model request fails, its error and validation questions are saved in a folder ending `_failed`, and testing continues with the next selection. Failed runs have no successful metrics and are excluded from comparisons. Successful benchmark exports are prepared under `results/.in_progress/` and moved into the normal results folder only after every file and graph is ready. Results scanning ignores in-progress folders.

Generated questions are balanced across farms and event/sensor/report types. When reports are enabled, the set includes repair and possible-cause questions with page-level references. They are derived from indexed content and **need human review before thesis reporting**. Scores on generated reports do not establish performance on independent real operational or maintenance questions.

Validation size controls generation size. Every displayed imported question runs. Relevant source IDs in the editable table use JSON lists:

```json
[
  {
    "id": "sensor-a-001",
    "question": "What does sensor_0 measure at Wind Farm A, and what is its unit?",
    "reference_answer": "Ambient temperature; unit °C",
    "relevant_ids": ["A:sensor:sensor_0"],
    "origin": "written and reviewed by researcher"
  }
]
```

Unknown source IDs and duplicate question IDs are rejected. Unanswerable questions can use `[]` and a reference of `Insufficient evidence.`. Export questions before changing the corpus, which clears the current validation set.

![Benchmark](docs/app-benchmark.png)

## 4. Results and comparisons

The app scans `results/` automatically on startup and every 15 seconds while idle. **Refresh saved runs** scans immediately.

1. Check the runs to compare.
2. Click **Compare selected runs**.
3. Select a graph metric and inspect the summary and replies.
4. Enter a comparison export name.
5. Click **Export comparison + graphs**.

Runs must have matching validation questions, corpus, embedding model, run mode, top-k, candidate count, temperature, and prompt version. Different answer models and algorithms can be compared. This prevents misleading averages across different experiments.

The report feature changes the answer prompt version and corpus. Start fresh benchmark runs for comparisons with reports; earlier CARE-only results are preserved and should be compared within their own compatible group.

Repeated runs are averaged per question first, then across questions. This is a descriptive comparison, not an automatic significance test. Unreadable files are skipped and counted. Old results and original data are preserved.

![Comparisons](docs/app-comparisons.png)

## Saved files

Each complete model/RAG test is named with Asia/Taipei date/time, LLM, and RAG. Microseconds prevent collisions. Characters unsuitable for folder names are replaced with hyphens.

```text
results/
  20261003_193000_123456_qwen2-5-7b_Hybrid/
    run.json
    results.csv
    summary.csv
    validation.json
    validation.csv
    corpus.json
    report.md
    graphs/
      recall_at_k.png
      precision_at_k.png
      mrr.png
      ndcg_at_k.png
      answer_f1.png
      exact_match.png
      citation_validity.png
      citation_recall.png
      total_ms.png
      retrieval_ms.png
      generation_ms.png
      quality_vs_latency.png
      per_question_metrics.png
  comparisons/
    20261003_194000_123456_my-comparison/
      comparison.json
      validation.json
      source_runs/              # Complete original run snapshots
      summary.csv
      results.csv
      report.md
      graphs/
```

`run.json` saves settings, outputs, metrics, evidence, reference answers, hashes, and package versions. Validation files preserve the exact questions. CSV evidence fields are JSON text. Each comparison records the source runs and exports all metric graphs and a quality-versus-latency plot.

The new PDF dataset lives separately:

```text
datasets/technician_reports/
  2023-01-27_Wind_Farm_A_Event_040_TR-A-040.pdf
  ...                                    # 45 separate two-page reports
  manifest.json
  README.md
```

The reproducible generator is `tools/generate_technician_reports.py`. Running it requires `reportlab` and `pandas`; the app itself needs only `pypdf` for PDF reading, included in `requirements.txt`. Install `reportlab` separately only if you want to regenerate the fixtures. The former farm-level compilations have been replaced by individual files. This changes source IDs and the prompt version; generate fresh benchmark results for compatible comparisons.

## Algorithms and metrics

| RAG | Retrieval |
| --- | --- |
| Dense | Cosine similarity over document/question vectors |
| Hybrid | Dense + BM25 keyword search, combined with reciprocal rank fusion |
| Hybrid + reranking | Hybrid candidates reordered by local LLM relevance scores |
| BM25 | Keyword retrieval using term frequency, rarity, and document length |
| TF-IDF | Cosine similarity over weighted words and two-word phrases |
| Dense + MMR | Dense candidate retrieval followed by maximal marginal relevance, balancing relevance and diversity |

**BM25** is a useful keyword baseline for explicit sensor and event names. **TF-IDF** is another lexical baseline that gives common words less weight. Both retrieve without calling the embedding model for the question; Ollama still generates their final answers. Their query embedding time is zero.

**MMR** means maximal marginal relevance. It first takes the strongest dense candidates, then repeatedly selects the source maximizing `0.7 × similarity to the question − 0.3 × maximum similarity to already selected sources`. This reduces repetitive context. The fixed relevance weight is recorded in each run. The candidate pool uses the **Reranking candidates** setting for both MMR and hybrid retrieval; it is always at least top-k. MMR scores are selection scores, not probabilities, and are not directly comparable to BM25 or cosine scores. Diversity may help some questions and reduce recall for others: benchmark it on the same reviewed questions.

All algorithms return up to top-k sources and use the same answer-generation prompt, so comparisons focus on the retrieval strategy. Query embedding time is charged only to algorithms that use dense retrieval. Index preparation time is recorded separately and excluded from question latency.

Ollama reranking uses a compact ordered numeric score list. Incomplete or invalid scores are explicit failures, not successful fallback results. Offline mode substitutes LSA vectors and character TF-IDF reranking and must be kept separate from real-model results.

| Metric | Direction | Meaning |
| --- | --- | --- |
| Recall@k | Higher is better ↑ | Fraction of relevant sources retrieved |
| Precision@k | Higher is better ↑ | Fraction of returned sources annotated relevant |
| MRR | Higher is better ↑ | Rewards finding the first relevant source early |
| nDCG@k | Higher is better ↑ | Rewards relevant evidence near the top |
| Answer token F1 | Higher is better ↑ | Word-token overlap with reference |
| Exact match | Higher is better ↑ | Normalized answer equality |
| Citation ID validity | Higher is better ↑ | Citation IDs occur in retrieved context |
| Citation recall | Higher is better ↑ | Relevant IDs are cited |
| Latency | Lower is better ↓ | Retrieval/generation/total milliseconds |

Charts include direction labels. Citation ID validity does **not** verify factual claim support, and answer F1 does **not** measure semantic truth. A correct answer can have noisy retrieval. Unanswerable questions have N/A retrieval scores. Per-run summaries include question-bootstrap intervals; aggregation graphs show descriptive means.

Document indexing is separate from per-question timing. Each sequential algorithm request embeds its question once. Local model loading and reranking can dominate time. Start with one model and a small validation set. Keep fixed settings consistent when comparing runs.

## Ollama

Install [Ollama for Windows](https://ollama.com/download/windows), then use installed embedding and chat models. For example:

```bat
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama list
```

If the service is not running, open Ollama or run `ollama serve`. The default address is `http://localhost:11434`. No local API key is required. Downloads require internet; local inference requires the service. The app does not automatically download models.

## Files and tests

- `app.py`: entry point.
- `windrag/workbench.py`: current chat, benchmark, and comparison interface.
- `windrag/ui.py`: shared setup/dataset/worker infrastructure.
- `windrag/data.py`: metadata, source IDs, unit repair, validation loading.
- `windrag/engine.py`: cached index, model requests, retrieval, generation.
- `windrag/evaluation.py`: metrics and run exports.
- `windrag/experiments.py`: sequential testing, run scanning, aggregation, graphs.
- `.cache/indexes/`: saved indexes.
- `tests/`: automated data, metrics, HTTP, cache, export, and UI checks.

Run in the activated environment:

```bat
python -m unittest discover -s tests -p "test_*.py" -v
python -c "import runpy; runpy.run_path('tests/smoke_workbench.py',run_name='__main__')"
```

The UI test uses offline mode with the actual upload, tests chat and three sequential algorithm runs, exports a comparison, and refreshes screenshots. Those saved runs are demonstration artifacts. HTTP tests use a deterministic server; real-model output still requires human review.

Troubleshooting: refresh models if lists are empty; load data before chatting; prepare references before benchmarking; select matching runs if aggregation rejects mixed settings. Smaller models or fewer candidates can reduce latency, but record those choices. The app answers historical records and does not predict faults or provide validated repair instructions.

