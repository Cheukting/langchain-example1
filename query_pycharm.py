import argparse
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_agent

from config import settings, require_api_key


def format_docs(docs) -> str:
    blocks: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "")
        blocks.append(f"Source: {src}\n---\n{d.page_content}")
    return "\n\n".join(blocks)


def build_agent(k: int = 6) -> AgentExecutor:
    """Create a minimal ReAct agent that can call a single retrieval tool."""
    require_api_key()

    # Load vector store and create retriever
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model, api_key=settings.openai_api_key
    )
    vector_store = FAISS.load_local(
        settings.index_dir, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k * 3, 12)}
    )

    # Wrap retriever as a v1 tool (function tool with decorator)
    @tool("pycharm_docs_search", return_direct=False)
    def pycharm_docs_search(q: str) -> str:
        """Search the local FAISS index of JetBrains PyCharm documentation and return relevant passages."""
        docs = retriever.get_relevant_documents(q)
        return format_docs(docs)

    # Simple ReAct-style system prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions about JetBrains PyCharm using the provided tools. "
                "Always consult the 'pycharm_docs_search' tool to find relevant documentation before answering. "
                "Cite sources by including the 'Source:' lines from the tool output when useful. If information isn't found, say you don't know.",
            ),
            ("human", "{input}"),
        ]
    )

    # Initialize the chat model via LangChain's init_chat_model helper
    llm = init_chat_model(
        model=settings.openai_chat_model,
        model_provider="openai",
        api_key=settings.openai_api_key,
        temperature=0,
    )
    # LangChain v1: use the generic create_agent factory
    agent = create_agent(llm=llm, tools=[pycharm_docs_search], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[pycharm_docs_search], verbose=False)
    return executor


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

    agent = build_agent(k=args.k)
    result = agent.invoke({"input": question})
    # AgentExecutor returns a dict with 'output' key
    print(result.get("output", ""))


if __name__ == "__main__":
    main()
