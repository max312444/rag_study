# 텍스트를 벡터로 바꿔주는 라이브러리
from sentence_transformers import SentenceTransformer
# 모델을 전역변수로 선언, 처음에는 비어있음
from config import EMBEDDING_MODEL
from typing import List

# 변수 앞에 _ 내부적으로만 쓰는 변수라는 관례
_model = None

def get_model() -> SentenceTransformer:
    # 함수 밖에 있는 _model 변수를 수정하겠다는 선언
    global _model
    # 모델이 없을 때만 로드
    if _model is None:
        print(f"모델 로딩 중: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("모델 로딩 완료")
    return _model

def embed(texts: List[str]) -> List[List[float]]:
    # 텍스트 리스트를 벡터 리스트로 변환
    model = get_model()
    # 10 -> 텍스트가 10개가 넘으면 진행바 표시
    vectors = model.encode(texts, show_progress_bar=len(texts) > 10)
    # numpy 배열을 python 리스트로 변환 (DB 저장용)
    return vectors.tolist()

def embed_one(text: str) -> List[float]:
    # 리스트로 감싸서 embed 호출. [0] -> 결과 리스트에서 첫 번째(유일한) 벡터만 꺼냄
    # 질문 하나 임베딩할 때 사용
    return embed([text])[0]

