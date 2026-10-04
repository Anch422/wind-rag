# LOUIE — concepts and research course

For the current app walkthrough, read [README.md](../README.md). This course explains the underlying concepts and how to interpret experiments.

### Chat history and model memory

LOUIE saves conversations locally using SQLite, a database included with Python. Each chat has its own messages, evidence, selected model, RAG algorithm, and draft. Saving messages as they arrive means you can close the application and return to the last chat. The sidebar lets you create, switch between, and delete chats.

Stored history and model memory are different features. LOUIE resolves chat follow-ups using user-mentioned farm, event/asset identifiers, and supported component names before retrieving evidence. An explicit new farm or an all-farms request replaces the inherited scope; each chat is isolated. The complete conversation is not sent to the model, and assistant statements do not determine inherited farm filters. Name the component explicitly when several issues are discussed. Benchmark questions bypass chat context and remain independent for fair comparison.

Chat history lives in `sessions/chat_history.sqlite3`, while research runs live in `results/`. Deleting a chat does not delete research results or PDF reports. Back up the database together with the dataset if you want to preserve conversations and working evidence links.

## 5. Python, terminals, and Conda explained

### Python and scripts

Python is the programming language used by the app. A `.py` file contains instructions. The Python interpreter reads those instructions and executes them. `app.py` is the entry point: it starts the desktop interface.

A **terminal** is a window where you type commands. Anaconda Prompt and PowerShell are different terminal/shell setups. The commands in this guide use Anaconda Prompt syntax; for example, `cd /d` belongs to that workflow.

Your **working directory** is the folder in which a command runs. If it does not contain `app.py`, `python app.py` cannot find the program. That is why you change folders first.

### Why an environment is useful

A Python project depends on a Python version and libraries. Different projects may need incompatible versions. A Conda environment is an isolated installation that gives this project its own Python and dependencies.

Think of the environment as a toolbox and the project folder as the documents you work on with that toolbox. They do not have to be stored together.

`base` is Anaconda's main environment. `wind-rag` is this project's named environment. Activating it changes which `python` and related tools your terminal uses; it does not move your files.

Useful checks:

```bat
conda env list
conda activate wind-rag
python --version
where python
python -m pip --version
```

After activation, `where python` should list the interpreter under `anaconda3\envs\wind-rag` first.

### Libraries, packages, dependencies, and installers

A **library** is reusable code. A **package** is a distributable unit of that code. A **dependency** is a package your program needs.

Conda manages Python and many compiled packages. `pip` installs Python packages, usually from PyPI. This project uses Conda for the basic scientific stack and can use pip for the GUI dependency. `python -m pip` selects the pip associated with the currently selected Python, reducing accidental installation into another environment.

`requirements.txt` lists the minimal application packages. `environment.yml` describes the broader research environment. Neither file downloads language-model weights; Ollama handles those separately.

Do not install packages repeatedly on every run. Install once, activate the environment, and start the app. Change dependencies deliberately if the program needs something new.

## 6. Understand your wind turbine dataset

### Wind turbines and SCADA

A wind turbine converts wind energy into electricity. Components include the rotor/blades, drivetrain, generator, pitch and yaw systems, electrical equipment, and control systems.

**SCADA** means Supervisory Control and Data Acquisition. Here it records operating measurements and status information, including sensor statistics at 10-minute intervals.

CARE to Compare was designed for fault/anomaly-detection research. This app uses its event metadata for historical question answering, which is a different research task.

### File layout

```text
datasets/CARE_To_Compare/
    README.md
    README.txt
    Wind Farm A/
        event_info.csv
        feature_description.csv
        datasets/
            0.csv
            ...
    Wind Farm B/
        ...
    Wind Farm C/
        ...
```

The large ZIP is the original archive. The app reads the extracted files and does not unpack the ZIP automatically.

### The three file types

| File | Contents | Use in this app |
| --- | --- | --- |
| `event_info.csv` | Event label, turbine asset, start/end, and optional root-cause description | One searchable document per event |
| `feature_description.csv` | Sensor names, meanings, units, and available statistics | One searchable document per sensor |
| `datasets/<event_id>.csv` | Timestamped SCADA measurements for one event dataset | On-demand sensor chart |

These CSVs use **semicolons** as separators. The loader handles UTF-8 and falls back to Latin-1 where necessary. Asset names differ between files: the loader accepts both `asset` and `asset_id`.

Actual uploaded metadata contains:

| Farm | Events | Anomaly | Normal | Sensor definitions |
| --- | ---: | ---: | ---: | ---: |
| A | 22 | 12 | 10 | 54 |
| B | 15 | 6 | 9 | 63 |
| C | 58 | 27 | 31 | 238 |
| Total | 95 | 45 | 50 | 355 |

The included dataset README describes an older 44/51 split. The app counts the actual event files rather than copying that older total. Dataset release notes explain label changes; keep the release version and metadata hash with your experiment.

### Labels and missing descriptions

`normal` and `anomaly` describe recorded events. A missing description is not evidence of a particular cause. The app represents missing normal descriptions as **Normal operation**, and missing anomaly descriptions as **Anomaly recorded; root cause not provided**.

Sensor-level statistics can include average, minimum, maximum, and standard deviation. The publisher documents quality issues in some minimum/maximum/standard-deviation signals; this app's chart selector focuses on average signals. Check the [dataset release notes](https://zenodo.org/records/15846963) before expanding sensor analysis.

The corpus includes historical event outcomes. Using them to answer retrospective questions is appropriate, but using those same outcomes as inputs to a purported early-fault prediction experiment would leak future information. This app makes no predictive-maintenance performance claim.

## 7. RAG explained from the beginning

### What a language model does

A language model generates text based on patterns learned during training and the input it receives. It does not automatically know the content of your uploaded files.

A model can produce a plausible answer without reliable evidence. RAG supplies relevant project-specific information before asking it to answer.

**RAG = Retrieval-Augmented Generation.**

- **Retrieval:** find source documents relevant to a question.
- **Augmentation:** put those documents into the model's input context.
- **Generation:** ask the model to answer using that evidence.

### The complete flow

```text
Metadata files -> source documents -> document embeddings -> searchable index
                                                               |
Question -> question embedding -> retrieval / fusion / reranking |
                                                               v
                           question + selected evidence -> local LLM -> answer + citations
```

In this project, each metadata record is already short enough to be a source document. The app does not split PDFs or long manuals into chunks. If you later add manuals, a chunking policy becomes another experimental choice.

### Corpus, document, index, and context

A **corpus** is the whole searchable collection. A **document** is one source unit within it. An **index** organizes the collection for search. The **context** is the selected evidence sent to the model for one question.

CARE contributes 450 documents; the supplied 90 report pages add chunks to that corpus. The interface defaults to top-k `3`, so at most three eligible sources enter one answer request.

### Embeddings and cosine similarity

An **embedding** is a list of numbers representing text. A neural embedding model learns representations that can place related text near one another even when the wording differs.

Cosine similarity compares vector directions. Larger similarity indicates greater alignment under that embedding model. It is a ranking signal, not a calibrated probability that the source is correct.

The app normalizes vectors and computes dot products, which become cosine similarities after normalization. At this corpus size, a NumPy matrix is sufficient; a separate vector-database service is not required.

### Prompts, tokens, and temperature

A **prompt** is the input sent to the model. It includes system instructions, the user's question, and the retrieved sources.

**Tokens** are the model's text-processing units. They are not always whole words. More context and longer answers can require more time and memory.

**Temperature** controls requested sampling randomness. Lower values encourage more consistent responses. The same prompt can still vary across execution environments.

The app requests concise answers, source-ID citations, and abstention when evidence is insufficient. These instructions help, but they do not guarantee factual accuracy.

### RAG versus fine-tuning and prediction

RAG retrieves external evidence at answer time. **Fine-tuning** changes a model's parameters through additional training. This app performs no fine-tuning.

### Technician-report evidence

The app can include searchable PDF reports alongside CARE metadata. Its default additional folder is `datasets/technician_reports`, containing 45 individual two-page reports. Generation history and verification status are preserved in the manifest, which is not sent to the LLM. Event mappings and recorded anomaly descriptions come from CARE; the generated service records have not been independently verified. Enable or disable reports in Setup to create combined-corpus or metadata-only experiments.

`pypdf` extracts page text; long pages become overlapping chunks. Source IDs retain PDF/page/chunk provenance. Repair and cause questions search report chunks, with an explicit farm narrowing the search. All six retrieval algorithms apply the same eligibility rule, allowing comparisons within that task. Problem questions search anomaly events; explicit normal-event requests search normal records, health overviews search both labels, and sensor questions use general retrieval. Explicit farms and event/asset identifiers narrow the same candidate pool for every algorithm. Generation is instructed to cite reports, describe the recorded repair directly, and keep suspected causes distinct from facts. This adds historical-report recommendation, not a validated diagnostic or autonomous maintenance system.

Report-based validation questions can be generated along with metadata questions. Review references and source coverage manually. Results on these generated fixtures should be described as a controlled RAG experiment, not evidence of real repair effectiveness. Scanned PDFs require OCR before import; the app does not perform OCR. Changing reports or the evidence prompt requires fresh compatible benchmark runs.

**Fault prediction** estimates a future operational outcome from observations. This app retrieves historical descriptions; answering a recorded-fault question does not demonstrate that a model could have predicted that fault beforehand.

## 8. How the six methods work

### Method 1 — Dense

1. Embed every source document once during indexing.
2. Embed the question.
3. Compare the question vector against all document vectors.
4. Select the top-k documents.
5. Generate a grounded answer.

Dense search can help with paraphrases, but exact sensor IDs, timestamps, and anonymized asset codes may be difficult for embeddings.

### Method 2 — Hybrid

Hybrid adds **BM25**, a keyword-ranking method. BM25 accounts for term frequency, rarity across documents, and document length.

The app combines the dense and keyword candidate lists using **reciprocal rank fusion (RRF)**:

```text
fused_score(document) = sum over lists of 1 / (60 + rank_in_list)
```

Ranks start at 1. Higher fused score is better. Only positive-scoring BM25 candidates contribute to its list. Fusion uses ranks because BM25 and cosine values do not have comparable numerical scales.

Hybrid can combine semantic matches with exact identifiers. It still depends on candidate depth and the embedding model.

### Method 3 — Hybrid + reranking

1. Run the same hybrid candidate retrieval.
2. Send the candidate texts and question to the selected local LLM.
3. Ask for a relevance score from 0 to 10 for each source ID.
4. Validate that the response contains exactly one finite, in-range score per candidate, in the supplied order.
5. Sort by those scores and select top-k sources.
6. Generate the answer using the same answer model as the other methods.

This is **LLM-based relevance reranking**, not a trained cross-encoder. It can improve selection but adds latency and may make unreliable relevance judgments. Invalid responses produce an explicit error; the app does not quietly call them successful reranked results.

### Method 4 — BM25

This uses the keyword-ranking component of Hybrid on its own. It requires saved term counts, document lengths, and inverse document frequencies. It does not embed the question. It is a lexical baseline for comparing semantic retrieval with exact word matching.

### Method 5 — TF-IDF

The app transforms the question into weighted word and two-word phrase features and ranks saved document TF-IDF vectors by cosine similarity. Frequent words contribute less than rare words. Unlike LSA, this baseline uses the original sparse word features without reducing their dimensions. No question embedding request is needed.

### Method 6 — Dense + MMR

Maximal marginal relevance (MMR) selects a diverse subset of dense candidates. At each step the app maximizes `0.7 × question similarity − 0.3 × maximum similarity to already selected documents`. It uses existing dense vectors and needs no additional document indexing. The candidate pool is controlled by the candidate setting. Diversity is a tradeoff: test whether it helps your question set rather than assuming it improves every answer.

### Preparing indexes and resetting Setup

Opening the app automatically discovers models, loads the dataset, prepares or reuses indexes, and scans saved results behind an initialization progress bar. Setup contains **General** and **RAG indexes** sub-tabs. Configure the dataset, run mode, and embedding model in General; changes schedule automatic preparation. All six algorithms share the saved dense, BM25, and TF-IDF components and become selectable after preparation succeeds. Restarting automatically loads the matching saved index, avoiding repeated document embeddings. Reranking and MMR happen at question time. **Reset to defaults** in General restores controls while preserving saved indexes and results. If initialization fails, correct Setup or select Offline demo; retry is available. **Refresh / retry indexes** is only needed for a manual refresh, such as edited metadata files on disk.

### Offline mode is a different experimental configuration

| Component | Ollama mode | Offline demonstration |
| --- | --- | --- |
| Dense representation | Neural embeddings | TF-IDF reduced with truncated SVD, also called LSA |
| Keyword retrieval | BM25 | BM25 |
| Fusion | RRF | RRF |
| Reranking | Local LLM relevance scoring | Character n-gram TF-IDF cosine |
| Answer | Local LLM using retrieved context | Top source's stored description and citation |

**TF-IDF** weights words by frequency and rarity. **SVD** reduces a large word-feature matrix to a lower-dimensional representation. **LSA** is this statistical latent representation; it is not a pretrained neural embedding model.

The offline answer is deliberately extractive. It copies the first retrieved record's answer field. It does not provide an independent generative reasoning test. Keep offline and Ollama results separate in your analysis.

## 9. Metrics: definitions and interpretation

Metrics require a reference question set. For each question, annotate an expected answer and the source IDs that supply the evidence.

### Retrieval metrics

Assume two documents are relevant and the app retrieves five documents, one of which is relevant.

| Metric | Definition | Example / interpretation |
| --- | --- | --- |
| Recall@k | Relevant documents retrieved / all annotated relevant documents | `1 / 2 = 0.5` |
| Precision@k | Relevant documents retrieved / documents retrieved | `1 / 5 = 0.2` |
| MRR | `1 / rank` of the first relevant document; zero if none retrieved | First relevant at rank 2 gives `0.5` |
| nDCG@k | Discounted relevance in the returned ranking divided by ideal ranking gain | Relevant evidence near the top scores better |

The implementation uses binary relevance, not graded relevance labels. Duplicate expected IDs are treated as a set.

If a question has no annotated relevant documents, these retrieval metrics are **N/A**, not perfect scores. Summary calculations exclude N/A values and record the number of evaluated questions per metric.

With one relevant source and top-k `5`, perfect recall can coexist with precision `0.2`. This is mathematically expected. Also, unannotated sources may contain useful evidence; incomplete relevance annotations can make measured precision misleading.

### Answer metrics

**Answer F1** compares normalized word tokens in the answer and reference. The evaluator removes bracketed citations, lowercases, removes punctuation through tokenization, and counts repeated token overlap.

```text
token precision = overlapping token count / predicted token count
token recall    = overlapping token count / reference token count
F1              = 2 * overlapping token count / (predicted count + reference count)
```

**Exact match** is 1 when the normalized answer equals the normalized reference, otherwise 0.

These measures are useful for short factual answers, but they penalize correct paraphrases and reward matching wording. They do not prove semantic correctness, completeness, causal reasoning, or safety. Read examples and add human assessment before making broad answer-quality claims.

### Citation metrics

- **Citation validity:** the fraction of bracketed citation IDs that occur among the retrieved sources.
- **Citation recall:** the fraction of annotated relevant source IDs that the answer cites.

No citations produce validity `0`. Citation recall is N/A for questions with no relevant sources.

An answer can cite a real retrieved ID and still misrepresent its content. **Citation validity is not faithfulness.** Claim-level support requires a separate human rubric or a validated judge; this app does not silently label ID checks as factual grounding scores.

### Time measurements

| Field | Includes |
| --- | --- |
| `index_build_ms` | Index construction; document embeddings in Ollama mode |
| `retrieval_ms` | Question embedding, ranking, fusion, and reranking when used |
| `generation_ms` | Answer generation request or offline extraction |
| `total_ms` | Retrieval and generation for one method/question |

When a comparison includes dense algorithms, each question is embedded once and the vector is shared. Its measured time is charged to each dense algorithm, so summed method times are not wall-clock duration. BM25 and TF-IDF have zero query embedding time. Index construction is reported separately. Chat runs only the selected algorithm; benchmarks generate every selected method's answer independently.

The first model request may include model loading. Hardware, active programs, model switching, context size, and server state affect latency. Repeat runs and document the setup before interpreting small timing differences.

Prompt and answer token counts are saved when Ollama supplies them. Reranking tokens are not added to answer token counts, so those fields are **not total pipeline token usage**. No monetary cost metric is computed for local inference.

### Confidence intervals

For each method and metric, the app resamples question-level scores with replacement 1,000 times and reports the 2.5th and 97.5th percentiles of their means.

These intervals describe variability within the supplied question set. They do not account for all model randomness, imperfect annotations, correlated questions, or differences between wind farms. They are not a significance test between methods. Identical easy-question scores can produce zero-width intervals.

## 10. Build a credible benchmark

### Separate learning, tuning, and final evaluation

Use starter questions to learn the app. Then write and review a benchmark that represents the questions your intended user would ask. Keep a development question set for tuning and a separate final set that you do not repeatedly tune against.

For RAG, the evidence needed to answer a test question may be in the searchable corpus. That is normal. The problem is using the final test questions or answers to tune your methods, writing trivial questions that expose their target IDs, or placing answers in prompts outside the retrieved context.

### Include multiple question types

- Exact lookup of an event by asset and time.
- Paraphrased questions about recorded component failures.
- Sensor-name and unit questions.
- Questions requiring evidence from more than one document.
- Ambiguous questions, with an explicit expected response.
- Questions the corpus cannot answer, with an abstention reference.

Do not ask for repair procedures if your corpus only contains fault labels. The reference should acknowledge missing evidence, or you should deliberately add trustworthy procedure documents in a future extension.

### Benchmark JSON format

```json
[
  {
    "id": "sensor-a-001",
    "question": "What does sensor_0 measure at Wind Farm A, and what is its unit?",
    "reference_answer": "Ambient temperature; unit °C",
    "relevant_ids": ["A:sensor:sensor_0"],
    "origin": "independently written and reviewed by researcher"
  },
  {
    "id": "unanswerable-001",
    "question": "What is the technician's phone number?",
    "reference_answer": "Insufficient evidence.",
    "relevant_ids": [],
    "origin": "independently written and reviewed by researcher"
  }
]
```

Save UTF-8 JSON under `benchmarks/`, then import it. Unique question IDs are required. Unknown evidence IDs are rejected.

CSV files need the same fields. For multiple relevant IDs, use a pipe-separated field or a JSON-list string. The app's **Export questions** creates a suitable CSV automatically.

### A reasonable first research workflow

1. Define the user task: historical operational question answering.
2. Write a research question, such as whether hybrid retrieval improves evidence recall over dense retrieval.
3. Create a small reviewed development set covering different farms and question types.
4. Choose one embedding model and one generator and keep them fixed across all six methods. With Hybrid + reranking, that generator also performs relevance scoring. Use the same reviewed validation set and settings for every run.
5. Use development questions to select top-k and candidate depth.
6. Freeze the settings and prepare an independent evaluation set.
7. Run all methods on that same set in Ollama mode.
8. Inspect failures, not only averages.
9. Repeat runs for timing/model variability; record hardware and model versions.
10. Report quality and latency together, with the limits of the corpus and annotations.

The app supports the runs and exports; it does not automatically establish independence, balance your questions, provide expert annotations, or run a paired significance test.

## 11. Saved results and reproducibility

Each successful benchmark creates a unique folder:

```text
results/<UTC-timestamp>-<short-id>/
    run.json
    results.csv
    summary.csv
    corpus.json
    comparison.png
```

| File | Purpose |
| --- | --- |
| `run.json` | Full run: settings, question set, per-question outputs, metrics, summary, hashes, versions, and notes |
| `results.csv` | One row per question/method, including answers and evidence |
| `summary.csv` | Mean metrics, evaluated-question counts, and confidence intervals |
| `corpus.json` | Source-document IDs and texts used in that run |
| `comparison.png` | Chart initially displayed after the benchmark |

Structured lists in CSV are JSON strings so evidence can be preserved in one cell. They are not lost merely to make the spreadsheet simpler.

A **SHA-256 hash** is a fingerprint of data. The app records corpus and question-set hashes, making it possible to check whether two runs used the same indexed content and questions.

For stronger reproducibility, also record:

- The CARE dataset release DOI/version.
- Ollama version, model tags and model digests.
- CPU/GPU, RAM, operating system, and whether models were already loaded.
- Any question-set review and changes between runs.
- Your source-code version and dependency versions.

The app records selected model names and package versions. Hardware is not recorded automatically. A model name alone is not an immutable model snapshot.

Large datasets, the ZIP archive, and generated run folders are ignored by Git through `.gitignore`. They remain on your computer. Back up selected thesis runs separately.

## 12. Files, libraries, and program architecture

### Organized project files

```text
_THESIS/
    app.py                  # Starts the application
    launch.bat              # Convenience launcher for a Conda-enabled terminal
    README.md               # This complete guide
    SETUP.md                # Short setup reference
    environment.yml         # Named Conda research environment
    requirements.txt        # Minimal app package list
    .cache/indexes/         # Saved indexes, shared by all six methods
    windrag/
        __init__.py
        data.py             # Dataset and benchmark loading
        engine.py           # Embeddings, retrieval, reranking, generation
        evaluation.py       # Metrics, summaries, run exports
        ui.py               # PyQt desktop interface and background tasks
    benchmarks/             # Your reviewed question sets
    results/                # Saved experiments
    docs/                   # Verified app screenshots
    tests/
        test_core.py        # Metric/data/model-contract tests
        test_ollama_http.py # Full HTTP pipeline against a local test server
        smoke_ui.py         # Offline end-to-end desktop workflow check
    datasets/               # Original extracted upload; preserved
    CARE_To_Compare.zip      # Original archive; preserved
```

### What each library does

| Library | Role |
| --- | --- |
| PyQt6 | Desktop windows, buttons, forms, tabs, tables, and signals |
| NumPy | Arrays, vector normalization/scoring, and numerical calculations |
| pandas | Read CSVs, work with event tables, aggregate sensor data, export results |
| scikit-learn | TF-IDF, SVD/LSA, and offline representations |
| Matplotlib | Embedded plots and saved PNG charts |
| requests | HTTP communication with the local Ollama service |
| joblib | Save and reload the local index, vector matrices, and fitted retrieval models |
| openpyxl | Available for later Excel work; not required by the app's CSV workflow |
| JupyterLab/ipykernel | Available for notebook exploration; not required to launch the desktop app |
| tqdm | Available for later terminal progress tools; the GUI uses its own progress indicator |

Python's standard library also provides JSON, paths, timestamps, hashing, randomization, and testing. These modules do not require separate installation.

The app does not require LangChain, FAISS, PyTorch, sentence-transformers, RAGAS, a paid API, or a vector-database server. Ollama handles local neural model inference outside Python.

### How the code fits together

`app.py` starts `windrag.ui.launch()`. The window creates a background worker for longer jobs. That worker calls the data loader, retrieval engine, or evaluator and sends progress/result signals back to the UI.

The **UI thread** handles buttons and screen updates. Worker threads handle indexing, chat requests, and sensor-file reading. Benchmarks run in a separate process using the same Conda Python, with a worker thread supervising progress. **Force cancel benchmark** terminates that process and closes its active HTTP connection immediately, without waiting for the response or timeout. Fully exported runs remain saved; the current incomplete export is discarded. Files are prepared in `results/.in_progress/` and moved into normal Results only when complete. Indexing, chat and sensor-reading cancellation remain cooperative.

`windrag/workbench.py` defines the current chat, model selectors, sequential benchmark, and comparison interface. `windrag/experiments.py` manages validation sampling, per-algorithm runs, charts, scanning, and comparison exports. The engine knows about source documents and models, but not buttons. The evaluator knows about answers and relevance labels, but not screen layout. Keeping those concerns separate makes it easier to extend or test the project.

### Run the checks

In Anaconda Prompt:

```bat
conda activate wind-rag
cd /d C:\Users\USER\Development\_THESIS
python -m unittest discover -s tests -p "test_*.py" -v
```

To run the full offline desktop smoke check:

```bat
python -c "import runpy; runpy.run_path('tests/smoke_ui.py', run_name='__main__')"
```

The smoke check loads the actual dataset, compares methods, saves a 12-question demo run, plots one sensor, and refreshes screenshots under `docs/`. It uses Qt's offscreen renderer and does not require a running Ollama server.

## 13. Troubleshooting

| Problem | What to do |
| --- | --- |
| `conda` is not recognized | Open Anaconda Prompt rather than an unconfigured terminal. |
| Environment not found | Run `conda env list`; create `wind-rag` if it is missing. |
| Python cannot find `app.py` | Change to `C:\Users\USER\Development\_THESIS` first. |
| `No module named PyQt6` | Activate `wind-rag`, then run `python -m pip install -r requirements.txt`. |
| No CARE metadata found | Select the extracted `CARE_To_Compare` folder, not the ZIP or its parent `datasets` folder. |
| Ollama connection refused | Open Ollama or start `ollama serve`, then check the host/port. |
| Model not found | Run `ollama list`, pull the model if needed, and use its exact tag in Setup. |
| HTTP 404 for `qwen2.5:3b` | This default model may not be installed. Click **Refresh models**, choose an installed chat model (for example your `qwen2.5:7b`), then rerun your question. Restart the app after code updates. |
| Embedding request fails | Confirm you selected an embedding model, not a chat model. |
| Reranker returns invalid scores | Retry one question, reduce candidate count, or use an answer model that reliably follows structured output; record changes and rerun. |
| Model request times out | Try fewer candidates or a smaller local model; verify the server is responding. |
| Index settings changed message | Setup changes prepare the matching index automatically. For edited files on disk, use Refresh / retry indexes. Answer-model and retrieval changes need no rebuild. |
| All methods score perfectly | Inspect whether the question set is trivial or metadata-generated; use reviewed paraphrases and varied evidence needs. |
| Precision is low but recall is high | Check top-k and annotation completeness; one relevant item among five gives precision 0.2. |
| Correct-looking answer has low F1 | Compare wording with the reference; token overlap does not fully capture semantic equivalence. |
| Cancel does not finish instantly | Wait for the current local-model request to complete or time out. |
| No result folder after an error | Incomplete runs are intentionally not saved as complete benchmarks. |

If a native Windows policy blocks a library, do not mistake that for a model-quality failure. Record the error and use an allowed installation/configuration. This app avoids the previously problematic FAISS/PyTorch Python stack.

## 14. Learning exercises

### Exercise A — Understand evidence

Ask a sensor question in offline mode. Find its citation in the source catalog. Read the source and verify the answer. This connects a generated-looking response to a real evidence record.

### Exercise B — Change top-k

Use the same reviewed questions with top-k `1`, `3`, and `5`, reusing the same index between settings. Compare recall, precision, and latency. Explain why increasing k can improve recall while decreasing precision.

### Exercise C — Test paraphrases

Write two versions of the same factual question: one using the exact recorded wording and another using a paraphrase. In Ollama mode, compare whether dense and hybrid retrieval behave differently. Keep their relevant IDs and reference answers consistent.

### Exercise D — Inspect reranking

Find a question where Hybrid and Hybrid + reranking retrieve different first sources. Read both source lists. Check whether the new ranking helps the answer or merely increases response time.

### Exercise E — Test missing knowledge

Write a question that the indexed records cannot answer. Annotate no relevant IDs and use `Insufficient evidence.` as the reference. Inspect the model's answer manually. Retrieval metrics are N/A here; correct abstention must be evaluated as an answer behavior.

### Exercise F — Write your methodology

Describe the corpus, question-set creation, fixed models, selected retrieval methods, settings, metrics, hardware, and limitations. Use the saved run files to support exact configuration details. Avoid using a perfect demo score as evidence of real-world operational capability.

## 15. Glossary and further reading

| Term | Meaning |
| --- | --- |
| API | Interface through which one program requests work from another |
| Asset | An anonymized turbine identifier in this dataset |
| Benchmark | A question set with reference answers and relevance annotations |
| BM25 | Keyword retrieval using frequency, rarity, and length normalization |
| Citation | Source ID attached to an answer |
| Conda environment | Isolated Python/tool installation |
| Corpus | Collection of source documents searched by the app |
| Dense retrieval | Search using numerical text representations |
| Embedding | Vector representation of text |
| Ground truth | Reviewed expected answer/evidence used for evaluation |
| Hallucination | Unsupported or incorrect generated information |
| Hybrid retrieval | Combination of vector and keyword search |
| Index | Data structures prepared for efficient retrieval |
| Inference | Running a trained model to produce an output |
| Latency | Time taken for an operation |
| LLM | Large language model |
| LSA | Latent semantic analysis using statistical dimension reduction |
| Metadata | Descriptive information about data, such as labels and units |
| RAG | Retrieval-augmented generation |
| Reranking | Reordering an initial retrieval candidate list |
| RRF | Reciprocal rank fusion |
| SCADA | Supervisory Control and Data Acquisition |
| Seed | Initial value controlling a pseudorandom process |
| Top-k | Number of highest-ranked sources selected |
| Vector | Ordered list of numbers |

Primary references:

- [CARE to Compare dataset and release notes](https://zenodo.org/records/15846963)
- [CARE to Compare research paper](https://arxiv.org/abs/2404.10320)
- [Conda environment management](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)
- [Ollama Windows installation](https://ollama.com/download/windows)
- [Ollama embeddings API](https://docs.ollama.com/api/embed)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [PyQt overview](https://riverbankcomputing.com/software)

When writing your thesis, cite the dataset and relevant original methods/papers. This README explains this implementation; it is not a substitute for a literature review.





