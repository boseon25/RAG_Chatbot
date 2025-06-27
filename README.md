# RAG Chatbot ������Ʈ

## ? ������Ʈ ����

�� ������Ʈ�� **BGE-M3 �Ӻ��� ��**, **FAISS ���� �����**, **Ollama-Exaone3.5 ����**�� Ȱ���� ���� ���� RAG(Retrieval-Augmented Generation) ê�� �ý����Դϴ�.

���� ���� PDF �������� ����ȭ�Ͽ� �����ϰ�, ������� ������ ���� ���� ������ ������� ��Ȯ�� �亯�� �����մϴ�.

## ? PDF ���� ���

`pdf/` �������� ������ ���� ���� ���� �������� ���ԵǾ� �ֽ��ϴ�:

### ? �����б��� ����
- **2022�����������������б��ȳ���.pdf**: 2022 ���� ���������� ���� �����б��� � ���̵����
- **2025�����б������ڷ�.pdf**: 2025�� �����б��� ��� ���� ���� �ڷ�

### ?? �б� ���� ����
- **2025���л��������հ�ȹ(�߼ۿ�).pdf**: 2025�� �л� ������ ���� �������� ��ȹ��
- **�б�������ȹ������������(����).pdf**: �б� ���� ��ȹ ������ ���� �������� ����

## ?? �ý��� ��Ű��ó

```
����� ���� �� BGE-M3 �Ӻ��� �� FAISS ���� �˻� �� ���� ���� ���� �� Ollama-Exaone3.5 �� �亯 ����
```

## ? �ڵ� ���� ����

### 1. ȯ�� ���� �� ���̺귯�� Import
```python
# �ʼ� ���̺귯��
from pathlib import Path
import numpy as np, requests
from FlagEmbedding import FlagModel
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
```

### 2. ���� ����
```python
PDF_PATH   = "pdf ���� ���"
INDEX_DIR  = "faiss_bge_m3_index"    # FAISS �ε��� ���� ���丮
CHUNK_SIZE = 800                     # �ؽ�Ʈ ûũ ũ��
CHUNK_OVER = 100                     # ûũ �� ��ġ�� �κ�
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "exaone3.5:2.4b"
```

### 3. BGE-M3 �Ӻ��� ���� Ŭ����
```python
class BGEEmbedding(Embeddings):
    """FlagEmbedding�� LangChain�� ȣȯ�ǵ��� ������ Ŭ����"""
    
    def __init__(self, model_name="BAAI/bge-m3", fp16=True):
        self.model = FlagModel(model_name, use_fp16=fp16)
    
    def _encode(self, texts):
        # �ؽ�Ʈ�� ���ͷ� ��ȯ�ϰ� ����ȭ
        vecs = self.model.encode(texts, batch_size=32, max_length=8192)
        vecs = np.asarray(vecs, dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs
    
    def embed_documents(self, texts): return self._encode(texts)
    def embed_query(self, text): return self._encode([text])[0]
```

### 4. FAISS �ε��� ���� �Լ�
```python
def build_faiss_index():
    """PDF ������ �ε��ϰ� FAISS ���� �ε����� ����/������Ʈ"""
    
    # PDF �ε�
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    
    # �ؽ�Ʈ ����
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVER
    )
    chunks = splitter.split_documents(docs)
    
    # FAISS �ε��� ���� �Ǵ� ������Ʈ
    if Path(INDEX_DIR).exists():
        # ���� �ε����� �߰�
        db = FAISS.load_local(INDEX_DIR, emb_wrapper, 
                              allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        # �� �ε��� ����
        db = FAISS.from_documents(chunks, emb_wrapper)
    
    # �ε��� ����
    db.save_local(INDEX_DIR)
    print(f"�ε��� �Ϸ� / �� ����: {db.index.ntotal:,}")
    return db
```

### 5. Ollama API ȣ�� �Լ�
```python
def ask_ollama(prompt: str, model: str = MODEL_NAME, max_tokens: int = 512):
    """Ollama API�� ���� Exaone3.5 �𵨿� ����"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    
    res = requests.post(OLLAMA_API, json=payload, timeout=120)
    if res.ok:
        return res.json().get("response", "").strip()
    return f"Ollama ����: {res.status_code} {res.text}"
```

### 6. RAG ���������� �Լ�
```python
def ask_rag(query: str, top_k: int = 3):
    """RAG ����������: �˻� �� ���ؽ�Ʈ ���� �� �亯 ����"""
    
    # 1. ���� �˻�
    db = FAISS.load_local(INDEX_DIR, emb_wrapper,
                          allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=top_k)
    
    # 2. ���ؽ�Ʈ ����
    context = "\n\n".join(d.page_content[:800] for d in docs)
    
    # 3. ������Ʈ ���� (�ҷ�ó��̼� ����)
    prompt = f"""�Ʒ� [����] ���뿡 �ٰ��ؼ��� �ѱ���� ���϶�.
������ ������ ������ '�𸣰ڽ��ϴ�'��� ���϶�.

[����]
{context}

[����]
{query}

[�亯]"""
    
    # 4. LLM ȣ��
    return ask_ollama(prompt)
```

## ? ��� ���

### 1. Ollama ��ġ �� �� �ٿ�ε�
```bash
# Ollama ��ġ (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Ollama ���� ����
ollama serve

# Exaone3.5 �� �ٿ�ε�
ollama pull exaone3.5:2.4b
```

### 2. Python ȯ�� ����
```bash
pip install FlagEmbedding langchain langchain-community faiss-cpu PyMuPDF requests numpy
```

### 3. ���� ����
```python
# FAISS �ε��� ���� (���� 1ȸ)
build_faiss_index()

# �����ϱ�
question = "�б�������� ���� ���� ü���� ��ȭ ������ ���� ���μ��� ���� ����ó�� ��� ��?"
answer = ask_rag(question)
print("Q:", question)
print("A:", answer)
```

## ? �ֿ� Ư¡

1. **���� ���� ����**: �ܺ� API ���� ���� ���ÿ��� ��� ó��
2. **���� ��� �亯**: �ҷ�ó��̼� ������ ���� ������ ������Ʈ ����
3. **���� �ε���**: �� ���� �߰� �� ���� �ε����� ���� ����
4. **ȿ������ �˻�**: FAISS�� ���� ��� ���� ���絵 �˻�
5. **�ѱ��� ����ȭ**: BGE-M3 �𵨰� Exaone3.5�� ���� �ѱ��� ó��

## ?? �ý��� �䱸����

- **Python**: 3.8 �̻�
- **�޸�**: 8GB RAM �̻� ����
- **�������**: �� �ٿ�ε�� 5GB �̻�
- **OS**: Windows, macOS, Linux

## ? �����ڷ�

- [BGE-M3 ��](https://huggingface.co/BAAI/bge-m3)
- [FAISS ���̺귯��](https://faiss.ai/)
- [Ollama](https://ollama.ai/)
- [LangChain](https://python.langchain.com/)

## ? ���� ����

- "�����б��� � �� ���ǻ����� �����ΰ���?"
- "�б� ������� �߻� �� �Ű� ������ ��� �ǳ���?"
- "�ߵ�� �λ� �߷��� ���� �̷��������?"
- "������� ���� ������ �� �ð� �ǽ��ؾ� �ϳ���?"

---

> **����**: �� �ý����� ������ �������� ���۵Ǿ�����, ���� ������ ��� �� �亯�� ��Ȯ���� �ݵ�� Ȯ���Ͻñ� �ٶ��ϴ�.
