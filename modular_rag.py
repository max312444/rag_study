import os
from dotenv import load_dotenv
from retriever import retrieve
from hybrid_retriever import hybrid_search
from reranker import rerank
from query_transformer import rewrite_query

load_dotenv()

# 각 모듈을 독립적인 함수로 정의
# Advanced RAG랑 달리 순서가 고정이 아니라 pipline_config로 조합 가능

def module_rewrite(query: str, api_key: str) -> str:
    print(f"  [모듈] 쿼리 재작성")
    return rewrite_query(query, api_key=api_key)

def module_vector_search(query: str, top_k: int) -> list:
    print(f"  [모듈] 벡터 검색")
    return retrieve(query, top_k=top_k)

def module_hybrid_search(query: str, top_k: int) -> list:
    print(f"  [모듈] 하이브리드 검색")
    return hybrid_search(query, top_k=top_k)

def module_rerank(query: str, chunks: list, top_k: int) -> list:
    print(f"  [모듈] 리랭킹")
    return rerank(query, chunks, top_k=top_k)

# 질문 유형을 판단하는 라우터
# LLM 없이도 동작하도록 키워드 기반으로 구현
def route_query(query: str) -> str:
    complex_keywords = ["비교", "차이", "설명", "자세히", "어떻게", "왜"]
    simple_keywords = ["뭐야", "뭔가요", "이란", "란"]

    for kw in complex_keywords:
        if kw in query:
            return "complex" # 재작성 + 하이브리드 + 리랭킹
        
    for kw in simple_keywords:
        if kw in query:
            return "simple" # 재작성 없이 벡터 검색만
        
    return "standard" # 재작성 + 벡터 검색 + 리랭킹


# 파이프라인 설정 - 어떤 모듈을 쓸지 딕셔너리로 정의
PIPELINE_CONFIGS = {
    "simple": {
        "rewrite": False,
        "search": "vector",
        "rerank": False,
    },
    "standard": {
        "rewrite": True,
        "search": "vector",
        "rerank": True,
    },
    "complex": {
        "rewrite": True,
        "search": "hybrid",
        "rerank": True,
    },
}

def modular_ask(query: str, api_key: str = None, top_k: int = 3):
    key = api_key or os.getenv("ANTHROPIC_API_KEY")

    print(f"\n=== [Modular RAG] 질문: {query} ===")

    # 1. 라우터가 질문 유형 판단
    query_type = route_query(query)
    config = PIPELINE_CONFIGS[query_type]
    print(f"\n[라우터] 유형: {query_type} | 설정: {config}")

    # 2. 설정에 따라 모듈 선택해서 실행
    search_query = query

    if config["rewrite"]:
        search_query = module_rewrite(query, key)
        print(f"  재작성: {search_query}")

    if config["search"] == "hybrid":
        chunks = module_hybrid_search(search_query, top_k=top_k * 2)
    else:
        chunks = module_vector_search(search_query, top_k=top_k * 2)

    if config["rerank"]:
        chunks = module_rerank(query, chunks, top_k=top_k)
    else:
        chunks = chunks[:top_k]
        for i , c in enumerate(chunks):
            c["rank"] = i + 1

    # 3. 결과 출력
    print(f"\n[최종 청크 {len(chunks)}개]")
    for r in chunks:
        score = r.get("rerank_score", r.get("score", 0))
        print(f"  #{r['rank']} | score={score:.4f} | {r['text'][:60]}...")

    context = "\n\n".join(r["text"] for r in chunks)
    print(f"\n[컨텍스트]\n{context[:300]}...")

if __name__ == "__main__":
    modular_ask("RAG란 무엇인가요?")
    modular_ask("벡터 검색과 BM25의 차이를 자세히  설명해줘")