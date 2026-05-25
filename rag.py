# 클로드 라이브러리
import anthropic
import os
from dotenv import load_dotenv
from setup_db import setup_db, setup_tables, reset_table
from indexer import index_file, list_indexed_docs
from retriever import retrieve, compare_methods

# API 키 관리
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")

def generate_answer(query: str, context: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    # 할루시네이션 방지용
    prompt = f"""다음 컨텍스트를 참고해서 질문에 답해주세요.
컨텍스트에 없는 내용은 모른다고 하세요.

컨텍스트:
{context}

질문: {query}
"""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def ask(query: str, api_key: str = None, top_k: int = 3):
    print(f"\n=== 질문: {query} ===")

    # 1. 유사 청크 검색
    results = retrieve(query, top_k=top_k)

    if not results:
        print("검색 결과 없음")
        return
    # 2. 검색 결과 출력
    print(f"\n[검색된 청크]")
    for r in results:
        print(f"\n  #{r['rank']} | 유사도: {r['score']:.4f} | {r['method']}")
        print(f"  {'-'*50}")
        print(f"  {r['text'][:150]}...")

    # 3. 컨텍스트 조합
    context = "\n\n".join(r["text"] for r in results)

    # 4. 답변 생성
    if api_key:
        print(f"\n[Claude 답변]")
        answer = generate_answer(query, context, api_key)
        print(answer)
    else:
        print(f"\n[API 키 없음 - 검색된 컨텍스트만 출력]")
        print(context)

if __name__ == "__main__":
    # DB 설정
    setup_db()
    setup_tables()


    # 샘플 문서 인덱승 (4가지 방법)
    FILE = "sample_docs/sample.txt"
    reset_table()
    # 같은 문서를 3가지 방식 다른 사이즈로 인덱싱
    # 나중에 compare_methods()로 어떤 방법이 더 좋은지 비교
    index_file(FILE, method="fixed", chunk_size=100) # 작은 사이즈
    index_file(FILE, method="fixed", chunk_size=300) # 큰 사이즈
    index_file(FILE, method="overlap", chunk_size=200, overlap=50)
    index_file(FILE, method="overlap", chunk_size=200, overlap=100) # overlap
    index_file(FILE, method="section")

    list_indexed_docs()

    # 질문 테스트
    ask("RAG란 무엇일까요?", api_key=API_KEY)

    print("\n=== 청킹 방법 비교 ===")
    results = compare_methods("딥러닝이란 무엇인가?")
    for method, chunks in results.items():
        print(f"\n[{method}]")
        for r in chunks:
            print(f"  유사도: {r['score']:.4f} | {r['text'][:80]}...")