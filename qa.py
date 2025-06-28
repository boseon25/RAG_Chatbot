import requests
from langchain_community.vectorstores import FAISS

from config import INDEX_DIR, OLLAMA_API, MODEL_NAME
from embedder import BGEEmbedding
from embedder import emb_wrapper

# Ollama-Exaone 호출 함수
# Ollama-Exaone 호출 함수
import re
import requests
from config import OLLAMA_API, MODEL_NAME

def ask_ollama(prompt: str, model: str = MODEL_NAME, max_tokens: int = 512):
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "options": {"num_predict": max_tokens}
    }

    try:
        res = requests.post(OLLAMA_API, json=payload, timeout=120)
        if res.ok:
            raw_response = res.json().get("response", "").strip()

            # 🔧 후처리: 특수문자 제거 (이모지, 전화기 기호 등)
            cleaned_response = re.sub(r"[☎📞📱]", "", raw_response)

            return cleaned_response
        else:
            return f"Ollama 오류: {res.status_code} {res.text}"
    except Exception as e:
        return f"요청 실패: {e}"


# RAG 파이프라인
# qa.py
def ask_rag(query: str, embedder, top_k: int = 3):
    from retriever import INDEX_DIR
    from langchain_community.vectorstores import FAISS

    db = FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=top_k)
    context = "\n\n".join(d.page_content[:800] for d in docs)

    prompt = f"""아래 [문서] 내용에 근거해서만 한국어로 답하라.
문서에 정보가 없으면 '모르겠습니다'라고 답하라.

[문서]
{context}

[질문]
{query}

[답변]"""

    return ask_ollama(prompt)
