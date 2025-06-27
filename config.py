from pathlib import Path

PDF_PATH   = r"C:\Users\user\Desktop\Github\RAG_Chatbot\pdf\2025년학생안전종합계획(발송용).pdf"
INDEX_DIR  = "faiss_bge_m3_index"
CHUNK_SIZE = 800      
CHUNK_OVER = 100
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "exaone3.5:2.4b"