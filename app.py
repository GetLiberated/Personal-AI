from flask import Flask, request, jsonify
from flask_cors import CORS
import os

gemini_api_key = os.environ.get('GOOGLE_API_KEY')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
pinecone_index = os.environ.get('PINECONE_INDEX')

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate

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


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say exactly "Sorry, I can't answer that.".
Use three sentences maximum and keep the answer as concise as possible.
Impersonate as a human named Eris and respond naturally.
If you are asked about your opinion, answer based on the context provided and your knowledge of Eris.
Don't greet the user, just answer the question directly.

{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def message_me():
    if request.method == 'GET':
        return 'ok'

    if not request.json:
        return jsonify({"error": "Invalid JSON"}), 400

    message = request.json.get('message')

    if not message:
        return jsonify({"error": "Message is required"}), 400

    response = graph.invoke({"question": message})

    return jsonify({"answer": response["answer"]})
