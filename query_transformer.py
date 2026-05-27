import anthropic
import os
import re
from dotenv import load_dotenv

load_dotenv()

def rewrite_query(query: str, api_key: str = None) -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return query
    
    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""다음 질문을 문서 검색에 최적화된 형태로 재작성하세요.
핵심 키워드를 명확하게 포함하고, 간결하게 작성하세요.
재작성된 질문만 출력하세요.

원래 질문: {query}"""
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        return query
    

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
                "content": f"""다음 질문을 3가지 다른 관점으로 재작성하세요.
각 질문은 <q> 태그로 감싸주세요.

원래 질문: {query}"""
            }]
        )
        queries = re.findall(r'<q>(.*?)</q>', response.content[0].text, re.DOTALL)
        result = [q.strip() for q in queries if q.strip()]
        return result if result else [query]
    except Exception:
        return [query]