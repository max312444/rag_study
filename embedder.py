from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
from typing import List

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"모델 로딩 중: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("모델 로딩 완료")
    return _model

def embed(texts: List[str]) -> List[List[float]]:
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=len(texts) > 10)
    return vectors.tolist()

def embed_one(text: str) -> List[float]:
    return embed([text])[0]
