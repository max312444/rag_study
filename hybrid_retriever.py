import psycopg2
from rank_bm25 import BM25Okapi
from config import DB_CONFIG
from embedder import embed_one

def get_all_chunks() -> list[dict]:
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
    query_vec = embed_one(query)

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
    tokenized_corpus = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    return [
        {**chunk, "score": float(score)}
        for score, chunk in ranked
    ]

def hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.6) -> list[dict]:
    chunks = get_all_chunks()

    vector_results = vector_search(query, top_k=top_k * 2)
    bm25_results = bm25_search(query, chunks, top_k=top_k * 2)

    # 각 방법의 최고점으로 정규화 (0~1 사이로 맞춤)
    def normalize(results):
        if not results:
            return {}
        max_score = max(r["score"] for r in results)
        if max_score == 0:
            return {r["id"]: 0.0 for r in results}
        return {r["id"]: r["score"] / max_score for r in results}
    
    vector_scores = normalize(vector_results)
    bm25_scores = normalize(bm25_results)

    # 두 점수를 가중 합산
    all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for cid in all_ids:
        v = vector_scores.get(cid, 0.0)
        b = bm25_scores.get(cid, 0.0)
        combined[cid] = vector_weight * v + (1 - vector_weight) * b

    # 점수 높은 순 정렬
    chunk_map = {c["id"]: c for c in chunks}
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