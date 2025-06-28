# -*- coding: utf-8 -*-
from embedder import BGEEmbedding
from retriever import build_faiss_index
from qa import ask_rag

def main():
    embedder = BGEEmbedding()
    db = build_faiss_index(embedder)
    q = "학교안전사고 예방 ·관리 체계의 고도화 과제에 대한 담당부서는 어디고 연락처가 어떻게 돼?"
    print("Q:", q)
    a = ask_rag(q, embedder)
    print("A:", a)

if __name__ == "__main__":
    main()