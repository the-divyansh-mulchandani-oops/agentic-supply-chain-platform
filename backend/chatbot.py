
import os
import sys
import uuid
import asyncio
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

warnings.filterwarnings("default")
sys.setrecursionlimit(10**7)

RECURSION_LIMIT = 10**7

SCRIPT_PATH = Path(__file__).resolve()
CURRENT_PATH = Path.cwd().resolve()

DB_PATH = Path(
    os.getenv("SQLITE_DB_PATH", "chat_memory.sqlite")
).expanduser()

if not DB_PATH.is_absolute():
    DB_PATH = CURRENT_PATH / DB_PATH

DB_PATH = DB_PATH.resolve()


def show(label, value):
    print(f"{label}: {value}")


async def chatbot_node(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


def build_graph(checkpointer):
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile(checkpointer=checkpointer)


async def print_history(graph, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])

    if not messages:
        show("History", "No messages in this conversation.")
        return

    for message in messages:
        if isinstance(message, HumanMessage):
            show("You", message.content)
        elif isinstance(message, AIMessage):
            show("Bot", message.content)


async def read_input(prompt):
    return await asyncio.to_thread(input, prompt)


async def main():
    global model

    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ),
        temperature=float(
            os.getenv("OLLAMA_TEMPERATURE", "0.7")
        ),
    )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    show("Current working directory", CURRENT_PATH)
    show("Script absolute path", SCRIPT_PATH)
    show("SQLite database absolute path", DB_PATH)
    show("Python recursion limit", sys.getrecursionlimit())
    show("LangGraph recursion limit", RECURSION_LIMIT)
    show("Warnings filter", "default")
    show("Ollama model", os.getenv("OLLAMA_MODEL", "llama3.1"))

    async with AsyncSqliteSaver.from_conn_string(
        str(DB_PATH)
    ) as checkpointer:
        graph = build_graph(checkpointer)
        thread_id = str(uuid.uuid4())

        show("Chat ID", thread_id)
        print(
            "Commands: /new, /switch <chat_id>, "
            "/history, /id, /exit"
        )

        while True:
            try:
                user_input = (
                    await read_input("\nYou: ")
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input == "/exit":
                print("Goodbye!")
                break

            if user_input == "/new":
                thread_id = str(uuid.uuid4())
                show("Started new chat", thread_id)
                continue

            if user_input == "/id":
                show("Chat ID", thread_id)
                continue

            if user_input == "/history":
                try:
                    await print_history(graph, thread_id)
                except Exception as error:
                    show("Error", f"{type(error).__name__}: {error}")
                continue

            if user_input.startswith("/switch "):
                new_id = user_input[len("/switch "):].strip()

                if new_id:
                    thread_id = new_id
                    show("Switched to chat", thread_id)
                else:
                    print("Usage: /switch <chat_id>")
                continue

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }

            try:
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                show("Bot", result["messages"][-1].content)

            except Exception as error:
                show("Error", f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
