from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVER, INDEX_DIR
from embedder import BGEEmbedding
from embedder import emb_wrapper

# 인덱스 생성 / 업데이트
def build_faiss_index(embedder):
    if Path(INDEX_DIR).exists():
        print("기존 인덱스 불러오는 중...")
        db = FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    else:
        print("새로 인덱싱 시작...")
        loader = PyMuPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER
        )
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(INDEX_DIR)
        print(f"인덱스 완료 / 총 벡터: {db.index.ntotal:,}")
    return db
