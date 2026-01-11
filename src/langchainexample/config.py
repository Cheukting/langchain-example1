import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    # Chat model assumed available per task
    openai_chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5")
    # Embedding model
    openai_embedding_model: str = os.environ.get(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
    )
    # Index directory
    index_dir: str = os.environ.get("INDEX_DIR", "indexes/pycharm_faiss")
    # Docs root
    docs_root: str = os.environ.get(
        "PYCHARM_DOCS_ROOT", "https://www.jetbrains.com/help/pycharm/"
    )


settings = Settings()


def require_api_key():
    if not settings.openai_api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in environment or .env file."
        )
