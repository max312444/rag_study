# python에서 PostgreSQL 연결하는 라이브러리
import psycopg2
#  DB 생성할 때 필요한 특수 모드
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import DB_CONFIG, EMBEDDING_DIM

def setup_db():
    # 1. DB 생성
    # 기본 DB에 먼저 접속. rag_study DB가 없을 수 있으니까 일단 기본 DB 먼저 접속
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        dbname="postgres",
    )
    # 일반 모드 -> 여러 쿼리를 묶어서 한번에 실행(트랜잭션)
    # 쿼리 하나하나 즉시 실행. DB 생성은 트랜잭션 안에서 못해서 이 모드가 필요
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # PostgreSQL이 관리하는 DB 목록 테이블. DB 없으면 생성하면 있으면 넘어감
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["dbname"],))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
        print(f"DB 생성 완료")
    else:
        print(f"DB 이미 존재")

    cur.close()
    conn.close()

def setup_tables():
    # 2. 테이블 생성
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # pgvector 확장 활성화
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 청크 저장 테이블
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id           SERIAL PRIMARY KEY,
            doc_name     TEXT NOT NULL,
            chunk_index  INTEGER NOT NULL,
            chunk_text   TEXT NOT NULL,
            chunk_method TEXT NOT NULL,
            embedding    vector({EMBEDDING_DIM})
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("테이블 생성 완료")

def reset_table():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()
    print("테이블 초기화 완료")

if __name__ == "__main__":
    setup_db()
    setup_tables()
    print("모든 설정 완료")