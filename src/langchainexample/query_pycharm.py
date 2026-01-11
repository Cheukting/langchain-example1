from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from dataclasses import dataclass

from .config import settings


@tool("pycharm_docs_search")
def pycharm_docs_search(q: str) -> str:
    """Search the local FAISS index of JetBrains PyCharm documentation and return relevant passages."""
    # Load vector store and create retriever
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model, api_key=settings.openai_api_key
    )
    vector_store = FAISS.load_local(
        settings.index_dir, embeddings, allow_dangerous_deserialization=True
    )

    k = 4

    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k * 3, 12)}
    )
    docs = retriever.invoke(q)

    def format_docs(docs) -> str:
        blocks: List[str] = []
        for d in docs:
            src = d.metadata.get("source", "")
            blocks.append(f"Source: {src}\n---\n{d.page_content}")
        return "\n\n".join(blocks)

    return format_docs(docs)


@dataclass
class ResponseFormat:
    """Response schema for the agent."""

    content: str


def main():
    parser = argparse.ArgumentParser(
        description="Ask PyCharm docs via an Agent (FAISS + GPT-5)"
    )
    parser.add_argument("question", type=str, nargs="+", help="Your question")
    parser.add_argument(
        "--k", type=int, default=6, help="Number of documents to retrieve"
    )
    args = parser.parse_args()
    question = " ".join(args.question)

    system_prompt = """You are a helpful assistant that answers questions about JetBrains PyCharm using the provided tools. 
    Always consult the 'pycharm_docs_search' tool to find relevant documentation before answering. 
    Cite sources by including the 'Source:' lines from the tool output when useful. If information isn't found, say you don't know."""

    agent = create_agent(
        model=settings.openai_chat_model,
        tools=[pycharm_docs_search],
        system_prompt=system_prompt,
        response_format=ToolStrategy(ResponseFormat),
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    print(result["structured_response"].content)


if __name__ == "__main__":
    main()
