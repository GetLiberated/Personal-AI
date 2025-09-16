from flask import Flask, request, jsonify
from flask_cors import CORS
import os

gemini_api_key = os.environ.get('GOOGLE_API_KEY')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_index = os.environ.get('PINECONE_INDEX')

# from langchain.globals import set_llm_cache
# from langchain_core.caches import InMemoryCache
# set_llm_cache(InMemoryCache())

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver


llm = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    google_api_key=gemini_api_key
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api_key
)


pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


system_message = """Impersonate as a human named Eris.
Only use the retrieve tool if the user's query is about you, Eris, that you would not know from your general training data.
If the user's query is a simple greeting, or a non-factual question, respond directly without using the tool.
Eris doesn't talk much, but he's low-key friendly.
If you don't have anything to say, say "Let me know if you have any other question." instead."""


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    messages_with_instructions = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages_with_instructions)
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say exactly \"Sorry, I'm not sure how to respond that.\"."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def message_me():
    if request.method == 'GET':
        return 'ok'

    if not request.json:
        return jsonify({"error": "Invalid JSON"}), 400

    message = request.json.get('message')
    thread = request.json.get('thread')

    if not message:
        return jsonify({"error": "Message is required"}), 400
    if not thread:
        return jsonify({"error": "Invalid thread ID"}), 400

    config = {"configurable": {"thread_id": thread}}
    response = graph.invoke({"messages": [{"role": "user", "content": message}]}, config=config)

    return jsonify({"answer": response["messages"][-1].content})
