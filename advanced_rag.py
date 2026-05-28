import os
from dotenv import load_dotenv
import anthropic
from query_transformer import rewrite_query, expand_query
from hybrid_retriever import hybrid_search
from reranker import rerank

load_dotenv()

def generate_answer(query: str, context: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""다음 컨텍스트를 참고해서 질문에 답해주세요.
컨텍스트에 없는 내용은 모른다고 하세요.

컨텍스트:
{context}

질문: {query}
"""
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def advanced_ask(query: str, api_key: str = None, top_k: int = 3):
    key = api_key or os.getenv("ANTHROPIC_API_KEY")

    print(f"\n=== [Advanced RAG] 질문: {query} ===")

    # 1. 쿼리 재작성
    rewritten = rewrite_query(query, api_key=api_key)
    print(f"\n[1] 쿼리 재작성: {rewritten}")

    # 2. 쿼리 확장
    expanded = expand_query(query, api_key=key)
    print(f"[2] 확장 쿼리 {len(expanded)}개: {expanded}")
    
    # 3. 하이브리드 검색 (재작성된 쿼리로)
    candidates = hybrid_search(rewritten, top_k=top_k * 3)
    print(f"\n[3] 하이브리드 검색 후보: {len(candidates)}개")
    for r in candidates:
        print(f"    score={r['score']:.4f} | {r['text'][:60]}...")

    # 4. 리랭킹
    final_chunks = rerank(query, candidates, top_k=top_k)
    print(f"\n[4] 리랭킹 후 최종: {len(final_chunks)}개")
    for r in final_chunks:
        print(f"    rerank={r['rerank_score']:.4f} | {r['text'][:60]}...")

    # 5. 답변 생성
    context = "\n\n".join(r["text"] for r in final_chunks)

    if key:
        print(f"\n[5] Claude 답변")
        answer = generate_answer(query, context, key)
        print(answer)
    else:
        print(f"\n[5] API 키 없음 - 최종 컨텍스트만 출력")
        print(context)

if __name__ == "__main__":
    advanced_ask("RAG란 무엇인가요?", api_key=os.getenv("ANTHROPIC_API_KEY"))