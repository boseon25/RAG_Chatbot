# RAG Chatbot

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/LangChain-Community-orange" alt="LangChain">
  <img src="https://img.shields.io/badge/BGE--M3-Embedding-purple" alt="BGE-M3">
  <img src="https://img.shields.io/badge/FAISS-VectorDB-red" alt="FAISS">
  <img src="https://img.shields.io/badge/Ollama-Exaone3.5-yellow" alt="Ollama">
</div>

<br>

<div align="center">
  <p><strong>? ���� ���� ���� ������ ������ RAG(Retrieval-Augmented Generation) ê��</strong></p>
  <p>BGE-M3 �Ӻ��� + FAISS ���� �˻� + Ollama-Exaone3.5 ������ Ȱ���� �ѱ��� ���� ��� �������� �ý���</p>
</div>

---

## Table of Contents

- [Features](#-features)
- [Architecture](#?-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Configuration](#?-configuration)
- [Development](#?-development)
- [Contributing](#-contributing)
- [License](#-license)

## Features

- **���� ���� ����**: �ܺ� API ���� ��� ó���� ���ÿ��� ����
- **PDF ���� ó��**: ���� ���� PDF ������ �ڵ����� ����ȭ�Ͽ� ����
- **��Ȯ�� �˻�**: FAISS ��� ��� ���� ���絵 �˻�
- **�ҷ�ó��̼� ����**: ���� ��� �亯�� �����ϴ� ������ ������Ʈ ����
- **�ѱ��� ����ȭ**: BGE-M3�� Exaone3.5�� ���� �ѱ��� Ưȭ ó��
- **���� �ε���**: �� ���� �߰� �� ���� �ε����� ȿ�������� ����

## Architecture

```mermaid
graph LR
    A[PDF ����] --> B[PyMuPDFLoader]
    B --> C[�ؽ�Ʈ ����]
    C --> D[BGE-M3 �Ӻ���]
    D --> E[FAISS ���� ����]
    
    F[����� ����] --> G[BGE-M3 �Ӻ���]
    G --> H[FAISS ���絵 �˻�]
    E --> H
    H --> I[���� ���� ����]
    I --> J[������Ʈ ����]
    J --> K[Ollama-Exaone3.5]
    K --> L[�亯 ����]
```

## Project Structure

```
RAG_Chatbot/
������ config.py              # ȯ�� ���� �� ��� ����
������ embedder.py            # BGE-M3 �Ӻ��� ���� Ŭ����
������ retriever.py           # FAISS �ε��� ���� �� �˻�
������ qa.py                  # Ollama API ȣ�� �� RAG ����������
������ main.py                # ���� ���� ��ũ��Ʈ
������ run_ollama.sh          # Ollama ��ġ �� ���� ��ũ��Ʈ
������ rag_final.ipynb        # Jupyter ��Ʈ�� (����/�׽�Ʈ��)
������ pdf/                   # PDF ���� ���� ����
��   ������ 2022�����������������б��ȳ���.pdf
��   ������ 2025���л��������հ�ȹ(�߼ۿ�).pdf
��   ������ 2025�����б������ڷ�.pdf
��   ������ 2025�ߵ���λ����ó�����1��.pdf
��   ������ 2025�ߵ���λ����ó�����2��.pdf
��   ������ [��ȹ]2025��������������������ͳݽ���Ʈ�����������汳���⺻��ȹ.pdf
��   ������ �б�������ȹ������������(����).pdf
������ faiss_bge_m3_index/    # FAISS ���� �ε��� (�ڵ� ����)
������ __pycache__/           # Python ĳ�� ����
```

### Core Modules

| ���� | ���� | �ֿ� ��� |
|------|------|-----------|
| `config.py` | ���� ���� | PDF ���, API ��������Ʈ, �� ���� �� |
| `embedder.py` | �Ӻ��� ó�� | BGE-M3 ���� LangChain�� ȣȯ�ǵ��� ���� |
| `retriever.py` | ���� �˻� | PDF �ε�, �ؽ�Ʈ ����, FAISS �ε��� ���� |
| `qa.py` | �������� | Ollama API ȣ��, RAG ���������� ���� |
| `main.py` | ���� ���� | ��ü �ý��� ���� �� ���� |

### Code Details

#### `config.py` - ȯ�� ����
```python
from pathlib import Path

PDF_PATH   = r"C:\path\to\pdf\file.pdf"  # PDF ���� ���
INDEX_DIR  = "faiss_bge_m3_index"        # FAISS �ε��� ���� ���丮
CHUNK_SIZE = 800                         # �ؽ�Ʈ ûũ ũ��
CHUNK_OVER = 100                         # ûũ �� ��ġ�� �κ�
OLLAMA_API = "http://localhost:11434/api/generate"  # Ollama API ��������Ʈ
MODEL_NAME = "exaone3.5:2.4b"            # ����� ��� ��
```

#### `embedder.py` - BGE-M3 �Ӻ��� ����
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
```

#### `retriever.py` - FAISS �ε��� ����
```python
def build_faiss_index(embedder):
    """PDF ������ �ε��ϰ� FAISS ���� �ε����� ����/������Ʈ"""
    if Path(INDEX_DIR).exists():
        # ���� �ε��� �ε�
        db = FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    else:
        # �� �ε��� ����
        loader = PyMuPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER
        )
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(INDEX_DIR)
    return db
```

#### `qa.py` - �������� �ý���
```python
def ask_rag(query: str, embedder, top_k: int = 3):
    """RAG ����������: �˻� �� ���ؽ�Ʈ ���� �� �亯 ����"""
    # 1. ���� �˻�
    db = FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=top_k)
    context = "\n\n".join(d.page_content[:800] for d in docs)
    
    # 2. ������Ʈ ���� (�ҷ�ó��̼� ����)
    prompt = f"""�Ʒ� [����] ���뿡 �ٰ��ؼ��� �ѱ���� ���϶�.
������ ������ ������ '�𸣰ڽ��ϴ�'��� ���϶�.

[����]
{context}

[����]
{query}

[�亯]"""
    
    # 3. LLM ȣ��
    return ask_ollama(prompt)
```

## ? Quick Start

### 1?? Prerequisites

```bash
# Python 3.8+ �ʿ�
python --version

# Git Ŭ��
git clone https://github.com/yourusername/RAG_Chatbot.git
cd RAG_Chatbot
```

### 2. Install Dependencies

```bash
# �ʼ� ��Ű�� ��ġ
pip install FlagEmbedding langchain langchain-community faiss-cpu PyMuPDF requests numpy
```

### 3. Setup Ollama

```bash
# Ollama ��ġ �� �� �ٿ�ε� (�ڵ�)
chmod +x run_ollama.sh
./run_ollama.sh
```

**Windows ������� ���:**
```powershell
# PowerShell���� ����
.\run_ollama.sh
```

### 4. Run the Chatbot

```bash
# ���� ��ũ��Ʈ ����
python main.py
```

## Usage

### Command Line Interface

```python
from embedder import BGEEmbedding
from retriever import build_faiss_index
from qa import ask_rag

# �Ӻ��� �� �ʱ�ȭ
embedder = BGEEmbedding()

# FAISS �ε��� ���� (���� 1ȸ)
db = build_faiss_index(embedder)

# �����ϱ�
question = "�б�������� ���� ���� ü���� ��ȭ ������ ���� ���μ��� ���� ����ó�� ��� ��?"
answer = ask_rag(question, embedder)

print(f"Q: {question}")
print(f"A: {answer}")
```

### Jupyter Notebook

`rag_final.ipynb` ������ ���� ��ȭ������ �׽�Ʈ�� �� �ֽ��ϴ�.

```bash
jupyter notebook rag_final.ipynb
```

## Dataset

### ���� ���� �÷���

| ī�װ� | ������ | ���� |
|----------|--------|------|
| **�����б���** | 2022�����������������б��ȳ��� | 2022 ���� �������� �����б��� ���̵� |
| | 2025�����б������ڷ� | 2025�� �����б��� � ���� �ڷ� |
| **�б�����** | 2025���л��������հ�ȹ | �л� ���� ���� ��ȹ�� |
| | �б�������ȹ������������ | ���� ��ȹ ���� ���� ���� |
| **�����λ�** | 2025�ߵ���λ����ó�����(1��) | �ߵ�� �λ���� ó�� ���̵� |
| | 2025�ߵ���λ����ó�����(2��) | �ߵ�� �λ���� ó�� ���̵� |
| **��������** | 2025����������������⺻��ȹ | ������� ���� �� ������ �ߵ� ���� |

## Configuration

### ȯ�� ���� ����

`config.py` ������ �����Ͽ� ������ ������ �� �ֽ��ϴ�:

```python
# PDF ���� ��� ����
PDF_PATH = r"C:\your\path\to\document.pdf"

# �ؽ�Ʈ ���� ũ�� ����
CHUNK_SIZE = 1000  # �� ū ûũ
CHUNK_OVER = 200   # �� ���� ��ħ

# �ٸ� �� ���
MODEL_NAME = "llama2:7b"  # �ٸ� Ollama ��
```

### BGE-M3 �Ӻ��� ����

```python
# GPU ��� �� (����)
embedder = BGEEmbedding(fp16=True)

# CPU�� ��� ��
embedder = BGEEmbedding(fp16=False)
```

## Development

### Testing

```bash
# ���� ��� �׽�Ʈ
python -c "from embedder import BGEEmbedding; print('Embedder OK')"
python -c "from retriever import build_faiss_index; print('Retriever OK')"
python -c "from qa import ask_rag; print('QA OK')"
```

### ���� ����͸�

```python
import time

start_time = time.time()
answer = ask_rag("your question here", embedder)
end_time = time.time()

print(f"���� �ð�: {end_time - start_time:.2f}��")
```

### Example Queries

```python
# ���� ���� ����
ask_rag("�б� ������� �߻� �� �Ű� ������?", embedder)

# �����б��� ���� ����  
ask_rag("�����б��� � �� ���ǻ�����?", embedder)

# ���� �λ� ���� ����
ask_rag("�ߵ�� ���� ��û �Ⱓ��?", embedder)

# �������� ���� ����
ask_rag("������� ���� ���� �ð���?", embedder)
```

### �ý��� �䱸����

| ������� | �ּ� �䱸���� | ���� ��� |
|----------|---------------|-----------|
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 10GB | 20GB+ |
| **GPU** | ���û��� | CUDA ���� GPU |
| **OS** | Windows/macOS/Linux | Any |

### Troubleshooting

#### �Ϲ����� ���� �ذ�

1. **Ollama ���� ����**
   ```bash
   # Ollama ���� �����
   ollama serve
   ```

2. **�޸� ���� ����**
   ```python
   # ûũ ũ�� ���̱�
   CHUNK_SIZE = 400
   ```

3. **�ѱ� ���ڵ� ����**
   ```python
   # UTF-8 ���ڵ� ���
   # -*- coding: utf-8 -*-
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- �ڵ� ��Ÿ��: PEP 8 �ؼ�
- Ŀ�� �޽���: ����� ��Ȯ�ϰ� �ۼ�
- ����ȭ: �ֿ� �Լ��� docstring �ۼ�
- �׽�Ʈ: �� ��� �߰� �� �׽�Ʈ �ڵ� ����

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[BGE-M3](https://huggingface.co/BAAI/bge-m3)**: ���� �ѱ��� �Ӻ��� ��
- **[FAISS](https://faiss.ai/)**: Facebook AI�� ���� ���絵 �˻� ���̺귯��
- **[Ollama](https://ollama.ai/)**: ���� LLM ���� �÷���
- **[LangChain](https://python.langchain.com/)**: LLM ���ø����̼� ���� �����ӿ�ũ
- **[Exaone3.5](https://www.lgresearch.ai/)**: LG AI Research�� �ѱ��� Ưȭ ����

---

<div align="center">
  <p>Made with ?? for Korean Education</p>
  <p>
    <a href="#-table-of-contents">Back to Top</a> ?
    <a href="https://github.com/yourusername/RAG_Chatbot/issues">Report Bug</a> ?
    <a href="https://github.com/yourusername/RAG_Chatbot/issues">Request Feature</a>
  </p>
</div>
