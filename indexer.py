import psycopg2
# 여러 행을 한번에 INSERT(일반 insert보다 훨씬 빠름)
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
        # zip -> 청크랑 벡터를 쌍으로 묶기 | enumerate -> 인덱스 번호 붙이기
        #  결과적으로 DB에 넣을 행 목록 생성
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    # rows 리스트를 한번에 INSERT
    execute_values(
        cur,
        """
        INSERT INTO chunks (doc_name, chunk_index, chunk_text, chunk_method, embedding)
        VALUES %s
        """,
        rows,
        # 리스트를 phvector 타입으로 변환
        template="(%s, %s, %s, %s, %s::vector)",
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f" 저장 완료")

def index_file(file_path: str, method: str = "section", **kwargs):
    # 파일 열고 읽기 (자동으로 닫힘). 한글 깨짐 방지까지
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    # 경로에서 파일 명만 추출
    doc_name = file_path.replace("\\", "/").split("/")[-1]
    index_document(doc_name, text, method=method, **kwargs)

def list_indexed_docs():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # COUNT(*) -> 청크 개수 세기.  GROUP BY -> 문서명 + 청킹방법 별로 묶기
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
    