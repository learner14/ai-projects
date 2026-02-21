from fastapi import FastAPI
import os
from langchain_openai import OpenAIEmbeddings
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain.chat_models import init_chat_model
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


os.environ["OPENAI_API_KEY"] = 'Your OpenAI API Key'
os.environ["ACTIVELOOP_TOKEN"] = 'ypur Activeloop API Key'

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings=OpenAIEmbeddings(model='text-embedding-ada-002')

db = DeeplakeVectorStore(dataset_path="./my_deeplake/", embedding_function=embeddings, overwrite=True)
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
     ),
    )
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
ids = db.add_documents(docs)
  
    # create Prompt template

#prompt = hub.pull("rlm/rag-prompt")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

 # define  state  for the Application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str



# Define application steps

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = db.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

tools = ToolNode([retrieve])

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
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
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


app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)


@app.get("/chat/")
async def root(query:str):
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

# Specify an ID for the thread
    config = {"configurable": {"thread_id": "abc123"}}


    for step in graph.stream(
    {"messages": [{"role": "user", "content": query}]}, stream_mode="values",config=config,):
        #step["messages"][-1].pretty_print()
        step["messages"][-1].pretty_print()
        print( "-------")

   # response = graph.invoke({"question": query})
   
    return {"test"}
