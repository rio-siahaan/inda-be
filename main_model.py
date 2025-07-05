from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_groq import ChatGroq
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv
import re
import uuid
import os

# Untuk Qdrant
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams


load_dotenv()

# Inisialisasi FastAPI
app = FastAPI()

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
client = QdrantClient(
        url="https://b91cf208-0d2a-4ec6-8711-78a6163bbc32.europe-west3-0.gcp.cloud.qdrant.io",
        api_key=os.getenv("QDRANT_API_KEY")
    )

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = QdrantVectorStore(
    client=client,
    collection_name="inda_collection",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
    vector_name="dense",
)

def remove_emojis(text):
    """Remove emojis from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_conversational_chain(selectedModel : str, persona: str, name: str):
    prompt_template = """
    Anda adalah INDA (Intelligent Data Assistant) yang menyediakan data dari BPS Provinsi Sumatera Utara yang hanya berfokus pada layanan penyediaan data. Anda juga dapat menjawab salam pembuka dan salam penutup selayaknya pelayan virtual.
    Jawablah dengan singkat dan akurat.
    Jika ada pertanyaan di luar konteks, Anda dapat meminta tahun dan variabel penjelas lainnya.
    Nama pengguna Anda adalah {name} dan personifikasi pengguna {persona}
    Berbahasalah sesuai informasi pengguna 
    Konteks:\n {context}\n
    Pertanyaan Pengguna: \n{input}\n    
    Jawaban yang relevan (berdasarkan dokumen):\n
    """

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 50})
    
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
            model="models/gemini-2.0-flash-lite",
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
        rag_chain = get_conversational_chain(selectedModel, persona, name)

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
            return "Maaf, layanan mencapai batas penggunaan (rate limit). Silakan coba lagi nanti."
        elif "503" in error_msg or "unavailable" in error_msg:
            return "Maaf, layanan sementara tidak tersedia. Mohon coba beberapa saat lagi."

        return f"Terjadi kesalahan: {str(e)}"

@app.post("/process_text")
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
