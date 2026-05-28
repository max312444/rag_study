import anthropic
import os
import re
from dotenv import load_dotenv

load_dotenv()

def rewrite_query(query: str, api_key: str = None) -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return query # 키 없으면 원본 질문 그대로 반환 (에러 안 남)
    
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                # 사용자 질문을 "검색하기 좋은 형태"로 바꿔달라고 LLM에 요청
                # 예) "RAG가 뭐야?" -> "RAG 개념과 동작 원리 설명"
                "content": f"""다음 질문을 문서 검색에 최적화된 형태로 재작성하세요.
핵심 키워드를 명확하게 포함하고, 간결하게 작성하세요.
재작성된 질문만 출력하세요.

원래 질문: {query}"""
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        return query # 키 없으면 원본 질문 1개짜리 리스트 반환
    

def expand_query(query: str, api_key: str = None) -> list[str]:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return [query]
    
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                # 같은 질문을 3가지 다른 관점으로 확장
                # 예) "딥러닝이란?" →
                #   "딥러닝 개념과 정의"
                #   "딥러닝과 머신러닝 차이"
                #   "딥러닝 활용 사례"
                "content": f"""다음 질문을 3가지 다른 관점으로 재작성하세요.
각 질문은 <q> 태그로 감싸주세요.

원래 질문: {query}"""
            }]
        )
        # LLM 응답에서 <q>...</q> 사이 텍스트만 추출
        queries = re.findall(r'<q>(.*?)</q>', response.content[0].text, re.DOTALL)
        result = [q.strip() for q in queries if q.strip()]
        return result if result else [query] # 태그 추출 실패하면 원본 반환
    except Exception:
        return [query]