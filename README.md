# LangChain PyCharm Docs QA (RAG Agent, LangChain v1 tool-calling)

This project builds a local retrieval-augmented generation pipeline to answer questions using the official PyCharm documentation at:
https://www.jetbrains.com/help/pycharm/

It uses [uv](https://docs.astral.sh/uv/) as project manager.

It uses:
- LangChain v1 tool-calling Agent for orchestration at query time
- OpenAI (GPT-5) as the chat model and `text-embedding-3-large` for embeddings
- FAISS as the local vector store
- Sitemap-based ingestion with a respectful fallback crawler

## Prerequisites

- Python 3.10+ (project file says 3.14+, but any recent 3.10/3.11/3.12 should work with these dependencies)
- An OpenAI API key

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Set environment variables (create a `.env` file or export in shell):
   ```
   OPENAI_API_KEY=sk-... # required
   OPENAI_CHAT_MODEL=gpt-5 # optional, default shown
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large # optional
   INDEX_DIR=indexes/pycharm_faiss # optional
   PYCHARM_DOCS_ROOT=https://www.jetbrains.com/help/pycharm/ # optional
   ```

## Build the index (ingest)

This downloads and processes the PyCharm docs and stores a FAISS index locally.
- Full run (can take some time):
  ```bash
  uv run ingest-pycharm
  ```
- Quick demo (limit number of pages):
  ```bash
  uv run ingest-pycharm --max-docs 120
  ```
The index is saved in `INDEX_DIR` (default `indexes/pycharm_faiss`).

## Ask questions (Agent)

Once indexing is complete, you can query with a LangChain v1 tool-calling Agent. The agent uses a single tool, `pycharm_docs_search`, which looks up relevant chunks in the local FAISS index and feeds them to GPT-5.
```bash
uv run ask-pycharm "How do I create a new project from existing sources?"
```

## Notes & Tips

- The system prompt instructs the model to answer only from the PyCharm docs context; when unsure, it should say it doesn't know.
- If you run into HTTP errors during ingestion, re-run later; the loader first tries the sitemap and then falls back to a shallow crawl.
- You can safely delete and rebuild the `indexes/` directory to refresh content.
- For better recall, you may experiment with `chunk_size`, `chunk_overlap`, or switch retrieval `search_type`.

## Entrypoints

- `ingest-pycharm` → `ingest_pycharm_docs.py:main`
- `ask-pycharm` → `query_pycharm.py:main` (Agent-based, tool-calling)

## License

MIT

## Disclaimer

This example project is generated with [Junie](https://www.jetbrains.com/junie/) and then examined and edited by a human. It is not intended for production use.
