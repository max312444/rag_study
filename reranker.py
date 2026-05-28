from sentence_transformers import CrossEncoder

_model = None # 전역 변수로 모델 캐싱 (embedding.py랑 같은 패턴)

def get_rerank_model() -> CrossEncoder:
    global _model
    if _model is None:
        print("리랭킹 모델 로딩 중...")
        # ms-marco: 검색 관련도 판단용으로 학습된 모델
        # MiniLM-L-6: 가볍고 빠른 버전
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("리랭킹 모델 로딩 완료")
    return _model

def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    if not chunks:
        return []
    
    model = get_rerank_model()

    # CrossEncoder는 (질문, 청크) 쌍을 같이 넣어서 관련도 점수를 직접 계산
    # 벡터 검색처럼 따로따로 임베딩하는 게 아니라 두 텍스트를 동시에 보기 때문에 더 정확
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs) # 각 쌍의 관련도 점수 반환

    # 점수 붙여서 높은 순 정렬
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # rerank_score 높은 순으로 정렬 후 top_k개만
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    for i, r in enumerate(reranked):
        r["rank"] = i + 1

    return reranked