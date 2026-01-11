import argparse
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from .config import settings, require_api_key


def format_docs(docs) -> str:
    blocks: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "")
        blocks.append(f"Source: {src}\n---\n{d.page_content}")
    return "\n\n".join(blocks)


def build_agent(k: int = 6) -> RunnableLambda:
    """Create a minimal tool-calling agent using bind_tools (LangChain v1)."""
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

    # Wrap retriever as a tool
    @tool("pycharm_docs_search")
    def pycharm_docs_search(q: str) -> str:
        """Search the local FAISS index of JetBrains PyCharm documentation and return relevant passages."""
        docs = retriever.invoke(q)
        return format_docs(docs)

    # Simple system prompt
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

    # Initialize the chat model
    llm = init_chat_model(
        model=settings.openai_chat_model,
        model_provider="openai",
        api_key=settings.openai_api_key,
        temperature=1,
    )

    # LangChain v1: Bind tools to the LLM
    llm_with_tools = llm.bind_tools([pycharm_docs_search])

    def agent_loop(data):
        input_text = data["input"]
        messages = prompt.format_messages(input=input_text)

        # First pass
        response = llm_with_tools.invoke(messages)

        # Handle tool calls (simple single-turn loop)
        if response.tool_calls:
            print("\n--- Action Plan ---")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(
                    f"{i}. Call tool '{tool_call['name']}' with args: {tool_call['args']}"
                )
            print("-------------------\n")

            messages.append(response)
            for tool_call in response.tool_calls:
                if tool_call["name"] == "pycharm_docs_search":
                    observation = pycharm_docs_search.invoke(tool_call["args"])
                    messages.append(
                        ToolMessage(content=observation, tool_call_id=tool_call["id"])
                    )

            # Get final answer after tool results are added
            response = llm_with_tools.invoke(messages)

        return {"output": response.content}

    return RunnableLambda(agent_loop)


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
