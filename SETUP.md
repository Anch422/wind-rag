# Start LOUIE

Read [README.md](README.md) for the complete beginner guide, local Ollama setup, concepts, and metrics.

The current UI uses **Chat** for one selected model/RAG, **Benchmark** for sequential model/algorithm tests, and **Results** to select saved runs and export comparisons. The longer theory course is in [docs/COURSE.md](docs/COURSE.md).

## First installation

Install Anaconda or Miniconda and enter the project folder in Anaconda Prompt. Run `conda create -n wind-rag python=3.11 pip`, activate it, then run `python -m pip install -r requirements.txt`. Skip creation if the environment exists. The named environment stays outside the project. GitHub excludes datasets, results, and chat sessions: supply CARE separately, and provide reports or disable them in Setup. See README for download, extraction, and report-generation instructions.

## Daily startup

Open the project folder containing `app.py` in File Explorer. Type `cmd` in the address bar and press Enter. Run:

```bat
conda activate wind-rag
python app.py
```

Then open **Ollama** from Start and leave it running. If LOUIE initialized before Ollama was available, click **Retry initialization**.

In VS Code choose **Terminal → Select Default Profile → Command Prompt**, open a new terminal in the project folder, and use the same two commands. If `conda` is not recognized, run `conda init cmd.exe` once in Anaconda Prompt, then reopen cmd. Keep the launching terminal open while LOUIE runs.

The app automatically initializes models, dataset, RAG indexes, and saved results when opened. Wait for the loading bar to finish. All six algorithms reuse the shared saved index across sessions. If Ollama is unavailable, Setup remains accessible: start Ollama and retry, or choose **Offline demo**, which initializes automatically. Answer models and retrieval settings can change without rebuilding. **Reset to defaults** restores setup choices while keeping saved indexes and results.

Technician PDF reports are enabled by default from `datasets/technician_reports`, with one report per PDF. Repair and possible-cause questions automatically retrieve report evidence with page citations. Generation history is preserved separately in the dataset manifest. To use your own searchable PDFs, choose another reports folder in Setup. Uncheck **Include technician PDF reports** for CARE-only experiments.

For real local RAG, install/start Ollama and pull the embedding and answer models listed in README.md. Select **Ollama** mode in the app.

If app libraries are missing, run `python -m pip install -r requirements.txt` after activating `wind-rag`.

## First benchmark

Select one answer model and all six RAGs. Generate 24 validation questions, review/edit their reference answers and relevant IDs, and export the reviewed set. Start sequential testing, then select the six compatible runs in Results and export the comparison. That produces 144 answers, six result folders, and comparison graphs. Offline demo checks the workflow; real LLM findings require Ollama and reviewed questions.

Chat history saves locally with New/Delete chat in the sidebar. Close normally to save drafts. Back up datasets, validation, `sessions/chat_history.sqlite3`, and `results/` separately from GitHub.

To recreate the environment:

```bat
conda env create -f environment.yml
```

