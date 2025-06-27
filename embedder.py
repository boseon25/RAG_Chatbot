from FlagEmbedding import FlagModel
from langchain.embeddings.base import Embeddings
import numpy as np

#  2) BGE-M3 임베딩 래퍼
class BGEEmbedding(Embeddings):
    """FlagEmbedding → LangChain compatible wrapper"""
    def __init__(self, model_name="BAAI/bge-m3", fp16=True):
        self.model = FlagModel(model_name, use_fp16=fp16)
    def _encode(self, texts):
        vecs = self.model.encode(texts, batch_size=32, max_length=8192)
        vecs = np.asarray(vecs, dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs
    def embed_documents(self, texts):  return self._encode(texts)
    def embed_query(self, text):       return self._encode([text])[0]

emb_wrapper = BGEEmbedding(fp16=True)      # GPU 없으면 fp16=False