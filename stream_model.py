from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
import re
import uuid
import os
import traceback

load_dotenv()

app = FastAPI()
set_llm_cache(InMemoryCache())
store = {}

class TextRequest(BaseModel):
    response_text: str
    id_chat: str = None
    selectedModel: str

client = QdrantClient(
    url="https://b91cf208-0d2a-4ec6-8711-78a6163bbc32.europe-west3-0.gcp.cloud.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="inda_collection",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
)

def clean_text(text):
    text = text.strip()

    text = re.sub(r'\s+', ' ', text.strip())
    
    text = re.sub(r'(?<=\w)\s(?=\W)', '', text)
    text = re.sub(r'(?<=\W)\s(?=\w)', '', text)
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_conversational_chain(selectedModel: str):
    prompt_template = """
    Anda adalah INDA (Intelligent Data Assistant) yang menyediakan data dari BPS Provinsi Sumatera Utara yang hanya berfokus pada layanan penyediaan data.
    Anda juga dapat menjawab salam pembuka dan salam penutup selayaknya pelayan virtual.
    Jawablah dengan ringkas dan akurat. Jika ada pertanyaan yang tidak jelas, Anda dapat meminta tahun dan variabel penjelas lainnya.
    Konteks:\n{context}\n
    Pertanyaan Pengguna:\n{input}\n
    Jawaban yang relevan (berdasarkan dokumen):
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})

    if selectedModel == "gemini":
        model = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash-lite",
            temperature=0.1,
            max_tokens=200,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            streaming=True,
        )
    elif selectedModel == "llama":
        model = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=200,
            groq_api_key=os.getenv("GROQ_API_KEY_1"),
            streaming=True,
        )
    else:
        raise ValueError("Model tidak dikenal")

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

@app.post("/process_text_stream")
async def process_text_stream(data: TextRequest):
    user_question = data.response_text
    id_chat = data.id_chat or str(uuid.uuid4())
    selected_model = data.selectedModel

    try:
        rag_chain = get_conversational_chain(selected_model)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        config = {"configurable": {"session_id": id_chat}}

        async def response_generator():
            full_text = ''
            async for chunk in conversational_rag_chain.astream(
                {"input": user_question}, config=config
            ):
                full_text += chunk.get("answer", '')
                cleaned_text = clean_text(full_text)
                print(cleaned_text)

                # Optional: Check if the text ends with a full stop to ensure it's finished
                if cleaned_text.endswith("."):
                    yield cleaned_text + " "
                else:
                    # Continue accumulating until the sentence is complete
                    pass

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        print("[‼️ ERROR DETECTED]", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Terjadi kesalahan: {str(e)}"},
        )
