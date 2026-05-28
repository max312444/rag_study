import psycopg2
from rank_bm25 import BM25Okapi
from config import DB_CONFIG
from embedder import embed_one

def get_all_chunks() -> list[dict]:
    # BM25는 DB에서 전체 텍스트를 다 가져와야 함
    # 벡터 검색은 DB가 알아서 계산하지만, BM25는 파이썬에서 직접 계산하기 때문
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT id, chunk_text, doc_name, chunk_method FROM chunks")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {"id": row[0], "text": row[1], "doc": row[2], "method": row[3]}
        for row in rows
    ]

def vector_search(query: str, top_k: int = 10) -> list[dict]:
    query_vec = embed_one(query) # 질문을 벡터로 변환

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
            SELECT id, chunk_text, doc_name, chunk_method,
                1 - (embedding <=> %s::vector) AS score 
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
    """, (query_vec, query_vec, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {"id": row[0], "text": row[1], "doc": row[2], "method": row[3], "score": float(row[4])}
        for row in rows
    ]

def bm25_search(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    # 전체 청크를 단어 단위로 쪼개서 BM25 색인 생성
    tokenized_corpus = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split() # 질문도 단어 단위로 쪼갬
    scores = bm25.get_scores(tokenized_query) # 각 청크의 BM25 점수 계산

    # 점수 높은 순으로 정렬해서 top_k개만 반환
    ranked = sorted(
        zip(scores, chunks), # 점수와 청크를 묶음
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [
        {**chunk, "score": float(score)} # 기존 청크 딕셔너리에 score 추가
        for score, chunk in ranked
    ]

def hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.6) -> list[dict]:
    chunks = get_all_chunks()

    # 두 방식으로 각각 후보를 넉넉하게 검색 (top_k * 2)
    vector_results = vector_search(query, top_k=top_k * 2)
    bm25_results = bm25_search(query, chunks, top_k=top_k * 2)

    def normalize(results):
        # 두 방식 점수 범위가 다르기 때문에 0~1로 맞춤
        # 벡터 점수: 0.1~0.9 / BM25 점수: 0~50 처럼 스케일이 달라서 합산 전에 필수
        if not results:
            return {}
        max_score = max(r["score"] for r in results)
        if max_score == 0:
            return {r["id"]: 0.0 for r in results}
        return {r["id"]: r["score"] / max_score for r in results}
    
    vector_scores = normalize(vector_results) # {청크id: 정규화점수} 딕셔너리
    bm25_scores = normalize(bm25_results)

    # 두 결과의 id를 합집합으로 모음
    all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for cid in all_ids:
        v = vector_scores.get(cid, 0.0) # 벡터 검색에 없으면 0점
        b = bm25_scores.get(cid, 0.0) # BM25 검색에 없으면 0점
        combined[cid] = vector_weight * v + (1 - vector_weight) * b # 벡터 60% + BM25 40%

    # 점수 높은 순 정렬
    chunk_map = {c["id"]: c for c in chunks} # id로 청크 빠르게 찾기 위한 딕셔너리
    results = sorted(
        [
            {**chunk_map[cid], "score": score}
            for cid, score in combined.items()
            if cid in chunk_map
        ],
        key=lambda x: x["score"],
        reverse=True
    )[:top_k]

    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results