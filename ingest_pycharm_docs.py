import argparse
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import settings, require_api_key


def load_docs(max_documents: int | None = None) -> List[Document]:
    base_url = settings.docs_root.rstrip("/") + "/"

    # Prefer sitemap to be polite and comprehensive; fallback to recursive crawl
    sitemap_url = base_url + "sitemap.xml"

    def is_article(url: str) -> bool:
        # Filter: Only help/pycharm pages; skip assets, indexes, pdf, anchors
        if not url.startswith(base_url):
            return False
        if any(
            url.endswith(ext)
            for ext in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".zip", ".pdf")
        ):
            return False
        return True

    docs: List[Document] = []
    try:
        loader = SitemapLoader(sitemap_url, filter_urls=[base_url])
        docs = loader.load()
    except Exception:
        # Fallback: gentle recursive crawl with limited depth
        loader = RecursiveUrlLoader(
            url=base_url,
            max_depth=2,
            use_async=True,
            timeout=20,
            prevent_outside=True,
            extractor=lambda soup: "\n".join(
                x.get_text(" ", strip=True) for x in soup.select("body")
            ),
        )
        docs = [d for d in loader.load() if is_article(d.metadata.get("source", ""))]

    # Deduplicate by source
    seen = set()
    unique_docs: List[Document] = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("loc") or ""
        if src and src not in seen and is_article(src):
            seen.add(src)
            unique_docs.append(
                Document(
                    page_content=d.page_content,
                    metadata={
                        "source": src,
                        **{k: v for k, v in d.metadata.items() if k not in {"source"}},
                    },
                )
            )

    if max_documents:
        unique_docs = unique_docs[:max_documents]
    return unique_docs


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", ".", "!", "?", ",", " "],
    )
    return splitter.split_documents(docs)


def build_index(max_documents: int | None = None) -> None:
    require_api_key()
    print("Loading PyCharm documentation...", flush=True)
    raw_docs = load_docs(max_documents=max_documents)
    print(f"Loaded {len(raw_docs)} pages. Splitting...", flush=True)
    chunks = split_docs(raw_docs)
    print(
        f"Created {len(chunks)} chunks. Embedding and building FAISS index...",
        flush=True,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model, api_key=settings.openai_api_key
    )
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(settings.index_dir)
    print(f"Index saved to {settings.index_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PyCharm docs into a FAISS index"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documentation pages for quick demo",
    )
    args = parser.parse_args()
    build_index(max_documents=args.max_docs)


if __name__ == "__main__":
    main()
