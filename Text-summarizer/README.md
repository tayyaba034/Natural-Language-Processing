# Text Summarizer

A simple web-based text summarization demo combining extractive and abstractive methods.

Features
- Extractive summarization using NLTK (sentence scoring by word frequency).
- Abstractive summarization using Hugging Face `transformers` pipeline (model: `sshleifer/distilbart-cnn-12-6`).
- Minimal Flask frontend served from `frontend.py` with a TailwindCSS-powered UI (template in `templates/index.html`).

Quick Start

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# transformers and a backend for the model (torch) are required for abstractive summarization:
pip install transformers torch
```

3. Run the app (starts a local Flask server):

```bash
python frontend.py
```

4. Open your browser to `http://127.0.0.1:5000/` and paste text to summarize.

Files
- `summarizer.py`: Extractive and abstractive summarization logic; includes a small `__main__` demo.
- `frontend.py`: Flask app exposing `/` (UI) and `/summarize` (POST JSON) endpoints.
- `templates/index.html`: Frontend UI using Tailwind CSS and client-side fetch to `/summarize`.
- `requirements.txt`: Minimal requirements (Flask, NLTK). See Quick Start for additional packages.

Notes & Troubleshooting
- On first run, NLTK data (punkt, stopwords) will be downloaded automatically by `summarizer.py`.
- The Hugging Face model will be downloaded on first abstractive request; this can take time and requires disk space and network access.
- For faster abstractive summarization, install a GPU-enabled PyTorch build and run on a machine with a CUDA GPU.
- If you only want extractive summaries, you can skip installing `transformers` and `torch` — the extractive functions work with just `nltk`.

Development
- To run the lightweight demo from the command line, execute:

```bash
python summarizer.py
```

License
- This repository contains example/demo code; add a license as needed.
