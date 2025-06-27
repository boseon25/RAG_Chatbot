# RAG Chatbot 프로젝트

## ? 프로젝트 개요

이 프로젝트는 **BGE-M3 임베딩 모델**, **FAISS 벡터 스토어**, **Ollama-Exaone3.5 언어모델**을 활용한 완전 로컬 RAG(Retrieval-Augmented Generation) 챗봇 시스템입니다.

교육 관련 PDF 문서들을 벡터화하여 저장하고, 사용자의 질문에 대해 문서 내용을 기반으로 정확한 답변을 제공합니다.

## ? PDF 문서 목록

`pdf/` 폴더에는 다음과 같은 교육 관련 문서들이 포함되어 있습니다:

### ? 자유학기제 관련
- **2022개정교육과정자유학기운영안내서.pdf**: 2022 개정 교육과정에 따른 자유학기제 운영 가이드라인
- **2025자유학기운영참고자료.pdf**: 2025년 자유학기제 운영을 위한 참고 자료

### ?? 학교 안전 관련
- **2025년학생안전종합계획(발송용).pdf**: 2025년 학생 안전을 위한 종합적인 계획서
- **학교안전계획직무연수교재(최종).pdf**: 학교 안전 계획 수립을 위한 직무연수 교재

## ?? 시스템 아키텍처

```
사용자 질문 → BGE-M3 임베딩 → FAISS 벡터 검색 → 관련 문서 추출 → Ollama-Exaone3.5 → 답변 생성
```

## ? 코드 구조 설명

### 1. 환경 설정 및 라이브러리 Import
```python
# 필수 라이브러리
from pathlib import Path
import numpy as np, requests
from FlagEmbedding import FlagModel
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
```

### 2. 설정 변수
```python
PDF_PATH   = "pdf 파일 경로"
INDEX_DIR  = "faiss_bge_m3_index"    # FAISS 인덱스 저장 디렉토리
CHUNK_SIZE = 800                     # 텍스트 청크 크기
CHUNK_OVER = 100                     # 청크 간 겹치는 부분
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "exaone3.5:2.4b"
```

### 3. BGE-M3 임베딩 래퍼 클래스
```python
class BGEEmbedding(Embeddings):
    """FlagEmbedding을 LangChain과 호환되도록 래핑한 클래스"""
    
    def __init__(self, model_name="BAAI/bge-m3", fp16=True):
        self.model = FlagModel(model_name, use_fp16=fp16)
    
    def _encode(self, texts):
        # 텍스트를 벡터로 변환하고 정규화
        vecs = self.model.encode(texts, batch_size=32, max_length=8192)
        vecs = np.asarray(vecs, dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs
    
    def embed_documents(self, texts): return self._encode(texts)
    def embed_query(self, text): return self._encode([text])[0]
```

### 4. FAISS 인덱스 생성 함수
```python
def build_faiss_index():
    """PDF 문서를 로드하고 FAISS 벡터 인덱스를 생성/업데이트"""
    
    # PDF 로드
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    
    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVER
    )
    chunks = splitter.split_documents(docs)
    
    # FAISS 인덱스 생성 또는 업데이트
    if Path(INDEX_DIR).exists():
        # 기존 인덱스에 추가
        db = FAISS.load_local(INDEX_DIR, emb_wrapper, 
                              allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        # 새 인덱스 생성
        db = FAISS.from_documents(chunks, emb_wrapper)
    
    # 인덱스 저장
    db.save_local(INDEX_DIR)
    print(f"인덱스 완료 / 총 벡터: {db.index.ntotal:,}")
    return db
```

### 5. Ollama API 호출 함수
```python
def ask_ollama(prompt: str, model: str = MODEL_NAME, max_tokens: int = 512):
    """Ollama API를 통해 Exaone3.5 모델에 질문"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    
    res = requests.post(OLLAMA_API, json=payload, timeout=120)
    if res.ok:
        return res.json().get("response", "").strip()
    return f"Ollama 오류: {res.status_code} {res.text}"
```

### 6. RAG 파이프라인 함수
```python
def ask_rag(query: str, top_k: int = 3):
    """RAG 파이프라인: 검색 → 컨텍스트 구성 → 답변 생성"""
    
    # 1. 벡터 검색
    db = FAISS.load_local(INDEX_DIR, emb_wrapper,
                          allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=top_k)
    
    # 2. 컨텍스트 구성
    context = "\n\n".join(d.page_content[:800] for d in docs)
    
    # 3. 프롬프트 생성 (할루시네이션 방지)
    prompt = f"""아래 [문서] 내용에 근거해서만 한국어로 답하라.
문서에 정보가 없으면 '모르겠습니다'라고 답하라.

[문서]
{context}

[질문]
{query}

[답변]"""
    
    # 4. LLM 호출
    return ask_ollama(prompt)
```

## ? 사용 방법

### 1. Ollama 설치 및 모델 다운로드
```bash
# Ollama 설치 (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Ollama 서버 실행
ollama serve

# Exaone3.5 모델 다운로드
ollama pull exaone3.5:2.4b
```

### 2. Python 환경 설정
```bash
pip install FlagEmbedding langchain langchain-community faiss-cpu PyMuPDF requests numpy
```

### 3. 실행 예시
```python
# FAISS 인덱스 생성 (최초 1회)
build_faiss_index()

# 질문하기
question = "학교안전사고 예방 관리 체계의 고도화 과제에 대한 담당부서는 어디고 연락처가 어떻게 돼?"
answer = ask_rag(question)
print("Q:", question)
print("A:", answer)
```

## ? 주요 특징

1. **완전 로컬 실행**: 외부 API 의존 없이 로컬에서 모든 처리
2. **문서 기반 답변**: 할루시네이션 방지를 위한 엄격한 프롬프트 설계
3. **증분 인덱싱**: 새 문서 추가 시 기존 인덱스에 누적 저장
4. **효율적인 검색**: FAISS를 통한 고속 벡터 유사도 검색
5. **한국어 최적화**: BGE-M3 모델과 Exaone3.5를 통한 한국어 처리

## ?? 시스템 요구사항

- **Python**: 3.8 이상
- **메모리**: 8GB RAM 이상 권장
- **저장공간**: 모델 다운로드용 5GB 이상
- **OS**: Windows, macOS, Linux

## ? 참고자료

- [BGE-M3 모델](https://huggingface.co/BAAI/bge-m3)
- [FAISS 라이브러리](https://faiss.ai/)
- [Ollama](https://ollama.ai/)
- [LangChain](https://python.langchain.com/)

## ? 예시 질문

- "자유학기제 운영 시 주의사항은 무엇인가요?"
- "학교 안전사고 발생 시 신고 절차는 어떻게 되나요?"
- "중등교원 인사 발령은 언제 이루어지나요?"
- "정보통신 윤리 교육은 몇 시간 실시해야 하나요?"

---

> **주의**: 이 시스템은 교육용 목적으로 제작되었으며, 실제 업무에 사용 시 답변의 정확성을 반드시 확인하시기 바랍니다.
