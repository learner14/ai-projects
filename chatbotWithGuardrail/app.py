from fastapi import FastAPI
import os

from langchain_openai import OpenAIEmbeddings
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = 'your_openai_api_key_here'
os.environ["ACTIVELOOP_TOKEN"] = 'your_activeloop_token_here'



app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

@app.get("/chat/")
def root(query:str):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
        ])
    output_parser = StrOutputParser()
    config = RailsConfig.from_path("config")
    guardrails = RunnableRails(config)

    chain = prompt | llm | output_parser
    chain_with_guardrails = guardrails | chain
    messages=  chain_with_guardrails.invoke({"input":query})
    if type(messages) is dict:
        response=messages['output']
    else:
        response=messages

    print(response)

    return {response}
