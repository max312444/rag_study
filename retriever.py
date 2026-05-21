import psycopg2
from config import DB_CONFIG
from embedder import embed_one

def retrieve(query: str, top_k: int = 3, method: str = None) -> list:
    # 1. 질문 임베딩
    query_vec = embed_one(query)

    # 2. DB에서 유사한 청크 검색
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    if method:
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

    rows = cur.fatchall()
    cur.close()
    conn.close()

    return [
        {"rank": i+1, "text": text, "doc": doc,
         "method": m, "score": float(score)}
         for i, (text, doc, m, idx, score) in enumerate(rows)
    ]

def compare_methods(query: str, top_k: int = 2) -> dict:
    results = {}
    for method in ["fixed", "overlap", "section"]:
        res = retrieve(query, top_k=top_k, method=method)
        if res:
            results[method] = res

    return results