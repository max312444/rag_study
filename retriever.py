import psycopg2
from config import DB_CONFIG
# 질문 하나를 벡터로 변환할 때 사용
from embedder import embed_one

# 질문을 벡터로 변환
def retrieve(query: str, top_k: int = 3, method: str = None) -> list:
    # 1. 질문 임베딩
    query_vec = embed_one(query)

    # 2. DB에서 유사한 청크 검색
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    if method:
        # <=> 거리 유사도로 변환
        # method 있으면 -> 특정 청킹 방법만 검색
        # method 없으면 -> 전체에서 검색
        cur.execute("""
            SELECT chunk_text, doc_name, chunk_method, chunk_index,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE chunk_method = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec, method, query_vec, top_k))
    else:
        cur.execute("""
            SELECT chunk_text, doc_name, chunk_method, chunk_index,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_vec, query_vec, top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # DB에서 가져온 결과를 딕셔너리 리스트로 변환
    # 나중에 r["score"], r["text"] 이렇게 꺼내 쓰기 편하게
    return [
        {"rank": i+1, "text": text, "doc": doc,
         "method": m, "score": float(score)}
         for i, (text, doc, m, idx, score) in enumerate(rows)
    ]

# 같은 질문을 3가지 청킹 방법으로 각각 검색
# 어떤 방법이 더 좋은 결과를 내는지 비교용
def compare_methods(query: str, top_k: int = 2) -> dict:
    results = {}
    for method in ["fixed", "overlap", "section"]:
        res = retrieve(query, top_k=top_k, method=method)
        if res:
            results[method] = res

    return results