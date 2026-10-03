# Start LOUIE

Read [README.md](README.md) for the complete beginner guide, local Ollama setup, concepts, and metrics.

The current UI uses **Chat** for one selected model/RAG, **Benchmark** for sequential model/algorithm tests, and **Results & comparisons** to select saved runs and export comparisons. The longer theory course is in [docs/COURSE.md](docs/COURSE.md).

In Anaconda Prompt:

```bat
conda activate wind-rag
cd /d C:\Users\USER\Development\_THESIS
python app.py
```

The app automatically initializes models, dataset, RAG indexes, and saved results when opened. Wait for the loading bar to finish. All six algorithms reuse the shared saved index across sessions. If Ollama is unavailable, Setup remains accessible: start Ollama and retry, or choose **Offline demo**, which initializes automatically. Answer models and retrieval settings can change without rebuilding. **Reset to defaults** restores setup choices while keeping saved indexes and results.

Technician PDF reports are enabled by default from `datasets/technician_reports`, with one report per PDF. Repair and possible-cause questions automatically retrieve report evidence with page citations. Generation history is preserved separately in the dataset manifest. To use your own searchable PDFs, choose another reports folder in Setup. Uncheck **Include technician PDF reports** for CARE-only experiments.

For real local RAG, install/start Ollama and pull the embedding and answer models listed in README.md. Select **Ollama** mode in the app.

If app libraries are missing, run `python -m pip install -r requirements.txt` after activating `wind-rag`.

For optional notebooks, run `jupyter lab` and select **Python (Wind RAG)**.

To recreate the environment:

```bat
conda env create -f environment.yml
```

