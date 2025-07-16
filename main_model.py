# from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv
import re
import uuid
import os

# Untuk Qdrant
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


load_dotenv()

# Inisialisasi FastAPI
# app = FastAPI()

# Setup cache Langchain
set_llm_cache(InMemoryCache())

# Setup penyimpanan riwayat percakapan
store = {}

# Pydantic input schema
class TextRequest(BaseModel):
    response_text: str
    id_chat: str 
    selectedModel: str
    persona: str
    name: str

# client = QdrantClient(path="/tmp/langchain_qdrant")
# client = QdrantClient(
#         url=os.getenv("QDRANT_URL"),
#         api_key=os.getenv("QDRANT_API_KEY"),
#         timeout=60
#     )

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="inda_collection",
#     embedding=embeddings,
#     retrieval_mode=RetrievalMode.DENSE,
#     vector_name="dense",
# )

def remove_emojis(text):
    """Remove emojis from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_conversational_chain(selectedModel : str, persona: str, name: str, vector_store):
    prompt_template = """
    Anda adalah INDA (Intelligent Data Assistant) yang menyediakan data dari BPS Provinsi Sumatera Utara yang hanya berfokus pada layanan penyediaan data. 
    Jawablah dengan ringkas dan akurat berdasarkan dokumen berikut.

    Nama pengguna: {name}
    Personifikasi pengguna: {persona}
    
    Konteks:\n {context}\n
    Pertanyaan Pengguna: \n{input}\n    
    Jawaban (berdasarkan dokumen):\n
    """

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", prompt_template),
    #         MessagesPlaceholder("chat_history"),
    #         ("human", "{input}")
    #     ]
    # )
    
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt_template),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]).partial(persona=persona, name=name)

    if selectedModel == "gemini":
        model = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.1,
            max_tokens=None,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif selectedModel == "llama":
        model = ChatGroq(
            temperature=0.1,
            model_name="llama-3.1-8b-instant",
            max_tokens=None,
            groq_api_key=os.getenv("GROQ_API_KEY_1")
        )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_response(user_question: str, session_id: str, selectedModel: str, persona: str, name: str):
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # Buat collection jika belum ada
        COLLECTION_NAME = "inda_collection"
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE)
                }
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )

        rag_chain = get_conversational_chain(selectedModel = selectedModel, persona = persona, name = name, vector_store = vector_store)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        config = {"configurable": {"session_id": session_id}}

        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config=config
        )

        response_text = response.get("answer", "")

        splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)  # token approx 1 word
        input_tokens = len(splitter.split_text(user_question))
        output_tokens = len(splitter.split_text(response_text))

        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        return remove_emojis(response_text), usage_metadata

    except Exception as e:
        error_msg = str(e).lower()

        # Deteksi Rate Limit / Quota Error
        if "rate limit" in error_msg or "quota" in error_msg or "429" in error_msg:
            return "Maaf, layanan mencapai batas penggunaan (rate limit). Silakan coba lagi nanti.", {}
        elif "503" in error_msg or "unavailable" in error_msg:
            return "Maaf, layanan sementara tidak tersedia. Mohon coba beberapa saat lagi.", {}

        return f"Terjadi kesalahan: {str(e)}", {}

# @app.post("/process_text")
async def process_text(data: TextRequest):
    response_text = data.response_text
    id_chat = data.id_chat or str(uuid.uuid4())
    selectedModel = data.selectedModel
    persona = data.persona
    name = data.name

    if not response_text:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No response text provided"})

    processed_text, usage_metadata = generate_response(response_text, id_chat, selectedModel, persona, name)
    return {
        "status": "success",
        "processed_text": processed_text,
        "id_chat": id_chat,
        "usage_metadata": usage_metadata
    }

# Di akhir file, pastikan hanya fungsi yang dipakai yang tersedia
# __all__ = ["generate_response"]