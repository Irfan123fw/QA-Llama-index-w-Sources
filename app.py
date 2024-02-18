import os
import chainlit as cl
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex,ServiceContext
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.langchain import LangChainLLM
load_dotenv()
import chromadb

# load from disk
embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
db2 = chromadb.PersistentClient(path="/path")
chroma_collection = db2.get_or_create_collection("langchain")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

@cl.on_chat_start
async def factory():

    llm=LangChainLLM(ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True))
    
    contents=f"Processing done "
    await cl.Message(content=contents).send()
    service_context = ServiceContext.from_defaults(
        llm=llm, 
        chunk_size=512, 
        chunk_overlap=20, 
        num_output=1000)
    
    memory = ChatMemoryBuffer.from_defaults(
        llm=llm, 
        token_limit=3000)
    
    query_engine = index.as_chat_engine(
        chat_mode="context", 
        service_context=service_context, 
        memory=memory,
        system_prompt=(        
        "Use the following pieces of information to answer the user's question." 
        "answer the user's question to the best of your ability using the resources provided."
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            ),
            )

    cl.user_session.set("query_engine", query_engine)

@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine") # type: ContextChatEngine
    response = await cl.make_async(query_engine.stream_chat)(message.content)
    response_message = cl.Message(content="")
    for token in response.response_gen:
        await response_message.stream_token(token=token)

    await response_message.send()
    
    label_list = []
    count = 1

    for sr in response.source_nodes:
        elements = [cl.Text(name="S"+str(count), content=f"{sr.node.text}", display="side", size='small')]
        response_message.elements = elements
        label_list.append("S"+str(count))
        await response_message.update()
        count += 1
    response_message.content += "\n\nSources: " + ", ".join(label_list)
    await response_message.update()
