import os
import pandas as pd
import streamlit as st
import io
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from API_GEMINI import GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv

load_dotenv()
# import faiss

def load_csv_files_with_metadata(csv_files):
    """Load CSV files, enrich with metadata, and return Document objects."""
    all_documents = []

    for file in csv_files:
        try:
            file_bytes = file.read()
            file.seek(0) 
            df = pd.read_csv(io.StringIO(file_bytes.decode('utf-8')))

            file_name = os.path.basename(file.name).split('.')[0]

            for index, row in df.iterrows():
                subject = str(row.get('label_subject', 'Subjek tidak diketahui'))
                vervar = str(row.get('label_vervar', 'Wilayah tidak diketahui'))
                var = str(row.get('label_var', 'Variabel tidak diketahui'))
                turvar = str(row.get('label_turvar', '')).strip()
                tahun = str(row.get('label_tahun', 'Tahun tidak diketahui'))
                nilai = str(row.get('nilai', 'Nilai tidak tersedia'))

                if turvar and turvar != "-":
                    content = (
                        f"Pada {turvar} tahun {tahun}, dalam topik '{subject}', indikator '{var}' untuk kategori '{vervar}' "
                        f"memiliki nilai sebesar {nilai}."
                    )
                else:
                    content = (
                        f"Pada tahun {tahun}, dalam topik '{subject}', indikator '{var}' untuk kategori '{vervar}' "
                        f"memiliki nilai sebesar {nilai}."
                    )


                metadata = {
                    "source": file_name,
                    "subject": subject,
                    "vervar": vervar,
                    "var": var,
                    "turvar": turvar if turvar != "-" else None,
                    "tahun": tahun,
                    "nilai": nilai,
                }

                document = Document(page_content=content, metadata=metadata)
                all_documents.append(document)

        except Exception as e:
            st.error(f"Gagal memproses file {file.name}: {e}")

    return all_documents


def create_or_update_vector_store(documents, batch_size):
    """Create or update a vector store with the given documents in batches, with progress bar."""
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    try:
        # Inisialisasi vector store
        if client.collection_exists("inda_collection"):
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="inda_collection",
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )
            st.info("📁 Koleksi vector store sudah ada. Melanjutkan update...")
        else:
            client.create_collection(
                collection_name="inda_collection",
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE)
                }
            )
            vector_store = QdrantVectorStore(
                client=client,
                collection_name="inda_collection",
                embedding=embeddings,
                retrieval_mode=RetrievalMode.DENSE,
                vector_name="dense",
            )
            st.success("🆕 Koleksi vector store baru berhasil dibuat.")

        # Mulai proses batching dengan progress bar
        total_docs = len(documents)
        total_batches = (total_docs + batch_size - 1) // batch_size
        success_count = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(0, total_docs, batch_size):
            batch_index = i // batch_size + 1
            batch_docs = documents[i:i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch_docs))]

            try:
                status_text.markdown(f"🔄 Memproses **batch {batch_index}/{total_batches}**: dokumen `{i}` hingga `{i + len(batch_docs) - 1}`")
                vector_store.add_documents(documents=batch_docs, ids=uuids)
                success_count += len(batch_docs)

                progress = success_count / total_docs
                progress_bar.progress(progress)
                st.success(f"✅ Batch {batch_index} berhasil ({len(batch_docs)} dokumen)")
            except Exception as e:
                st.error(f"❌ Gagal di batch {batch_index}: {e}")
                break

        if success_count == total_docs:
            st.balloons()
            st.success(f"🎉 Semua dokumen berhasil dimasukkan ({success_count}/{total_docs})")
        else:
            st.warning(f"⚠️ Proses berhenti. Total berhasil dimasukkan: {success_count}/{total_docs}")

    except Exception as e:
        st.error(f"🚫 Error saat membuat atau memperbarui vector store: {e}")
        return None

    return vector_store


def get_conversational_chain():
    """Create and return a QA chain."""
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) pada aplikasi WhatsApp yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan permintaan data khususnya dari BPS Provinsi Sumatera Utara. Sebagai kaki tangan BPS Provinsi Sumatera Utara, Anda tidak boleh mendiskreditkan BPS Provinsi Sumatera Utara. Kepala BPS Provinsi Sumatera Utara adalah Asim Saputra, SST, M.Ec.Dev. Kantor BPS Provinsi Sumatera Utara berlokasi di Jalan Asrama No. 179, Dwikora, Medan Helvetia, Medan, Sumatera Utara 20123.

    Visi BPS pada tahun 2024 adalah menjadi penyedia data statistik berkualitas untuk Indonesia
    Maju.
    Misi BPS pada tahun 2024 meliputi: 1) Menyediakan statistik berkualitas yang berstandar
    nasional dan internasional; 2) Membina K/L/D/I melalui Sistem Statistik Nasional yang
    berkesinambungan; 3) Mewujudkan pelayanan prima di bidang statistik untuk terwujudnya
    Sistem Statistik Nasional; 4) Membangun SDM yang unggul dan adaptif berlandaskan nilai
    profesionalisme, integritas, dan amanah.

    Anda tidak menerima input berupa audio dan gambar.

    Jawaban Anda harus sesuai dengan konteks dan tidak memberikan informasi yang salah atau di luar konteks. Jika ada permintaan data di luar konteks, arahkan pengguna ke https://sumut.bps.go.id untuk informasi lebih lanjut.
    
    Context:\n {context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    """Handle user input and get response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
    # client = QdrantClient(path="/tmp/langchain_qdrant")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )
    # sparse_embeddings = FastEmbedSparse(model_name = "Qdrant/bm25")
    
    try:
        # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        new_db = QdrantVectorStore(
            client=client, 
            collection_name="inda_collection", 
            embedding=embeddings, 
            # sparse_embedding = sparse_embeddings, 
            retrieval_mode=RetrievalMode.DENSE, 
            vector_name = "dense", 
            # sparse_vector_name = "sparse"
            )

        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("Reply:", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing user input: {e}")

def main():
    st.set_page_config(page_title="Chat CSV")
    st.header("Chat with CSV using Gemini")
    
    user_question = st.text_input("Ask a Question from the CSV Files")
    
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.title("Menu: ")
        
        if 'csv_files' not in st.session_state:
            st.session_state.csv_files = []

        if 'documents' not in st.session_state:
            st.session_state.documents = []

        if 'current_batch' not in st.session_state:
            st.session_state.current_batch = 0

        if 'batch_size' not in st.session_state:
            st.session_state.batch_size = 25

        uploaded_files = st.file_uploader("Upload your CSV Files", accept_multiple_files=True)

        if uploaded_files:
            st.session_state.csv_files = uploaded_files
            st.session_state.documents = load_csv_files_with_metadata(uploaded_files)
            st.success(f"📦 {len(st.session_state.documents)} dokumen berhasil diparsing.")


        # Hitung total batch
        total_docs = len(st.session_state.documents)
        total_batches = (total_docs + st.session_state.batch_size - 1) // st.session_state.batch_size

        st.info(f"🧮 Total dokumen: {total_docs}, total batch: {total_batches}")

        auto_process = st.sidebar.checkbox("🔁 Proses Otomatis Setelah Upload", value=True)

        if total_docs > 0 and st.session_state.current_batch == 0 and auto_process:
            vector_store = create_or_update_vector_store(
                documents=st.session_state.documents,
                batch_size=st.session_state.batch_size
            )
            st.session_state.current_batch = total_batches

        # Proses batch per tombol
        if st.button("▶️ Proses Batch Selanjutnya"):
            if st.session_state.current_batch < total_batches:
                start = st.session_state.current_batch * st.session_state.batch_size
                end = start + st.session_state.batch_size
                batch_docs = st.session_state.documents[start:end]

                st.write(f"📤 Memproses batch {st.session_state.current_batch + 1} dari {total_batches} (dokumen {start}-{end})")

                try:
                    # Kirim ke vector store
                    create_or_update_vector_store(batch_docs, batch_size=len(batch_docs))  # batch kecil satu kali
                    st.success(f"✅ Batch {st.session_state.current_batch + 1} berhasil.")
                    st.session_state.current_batch += 1

                    # Tampilkan progres
                    st.progress(st.session_state.current_batch / total_batches)

                    if st.session_state.current_batch == total_batches:
                        st.balloons()
                        st.success("🎉 Semua batch berhasil diproses.")
                except Exception as e:
                    st.error(f"❌ Gagal memproses batch {st.session_state.current_batch + 1}: {e}")
            else:
                st.warning("✅ Semua batch sudah diproses.")
                
if __name__ == "__main__":
    main()
