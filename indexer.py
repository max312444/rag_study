import psycopg2
from psycopg2.extras import execute_values
from config import DB_CONFIG
from chunking import chunk_text
from embedder import embed

def index_document(doc_name: str, text: str, method: str = "section", **kwargs):
    print(f"\n[인덱싱] '{doc_name}' | 방법: {method}")

    # 1. 청킹
    chunks = chunk_text(text, method=method, ** kwargs)
    print(f"  청크 수: {len(chunks)}")

    # 2. 임베딩
    print(" 임베딩 생성 중...")
    vectors = embed(chunks)

    # 3. DB 저장
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    rows = [
        (doc_name, i, chunk, method, vector)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    execute_values(
        cur,
        """
        INSERT INTO chunks (doc_name, chunk_index, chunk_text, chunk_method, embedding)
        VALUES %s
        """,
        rows,
        template="(%s, %s, %s, %s, %s::vector)",
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f" 저장 완료")

    def index_file(file_path: str, method: str = "section", **kwargs):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        doc_name = file_path.replace("\\", "/").split("/")[-1]
        index_document(doc_name, text, method=method, **kwargs)

    def list_indexed_docs():
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
    cur.execute("""
        SELECT doc_name, chunk_method, COUNT(*) as chunk_count
        FROM chunks
        GROUP BY doc_name, chunk_method
        ORDER BY doc_name, chunk_method
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print("\n[인덱싱된 문서]")
    if not rows:
        print("  (없음)")
    for doc_name, method, count in rows:
        print(f"  {doc_name} | {method:10s} | {count}개 청크")
        