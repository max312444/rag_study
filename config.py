# 기본 DB 설정 로케이션 설정
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5434,
    "dbname": "rag_study",
    "user": "postgres",
    "password": "postgres",
}

# 모델 설정
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# 모델이 만드는 벡터의 크기. 숫자 384개짜리 배열로 변환
EMBEDDING_DIM = 384