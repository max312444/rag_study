# 정규표현식 라이브러리, 문장 분리할 때 사용
import re
# 함수가 리스트를 반환한다고 표시해주는 타입 힌트
from typing import List

# 글자 수 기준 청킹(문맥을 딱 끊어서 하는 방법)
def fixed_chunking(text: str, chunk_size: int = 200) -> List[str]:
    chunks = []
    # 텍스트 맨 앞에서 부터 시작
    start = 0
    while start < len(text):
        # 청크 사이즈 만큼이 끝점
        end = start + chunk_size
        # 리스트에 시작~끝 잘라서 추가
        chunks.append(text[start:end])
        # 다음 청크는 바로 다음부터 시작
        start = end
    return [c for c in chunks if c.strip()]

def overlap_chunking(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # fixed랑 다른 부분(청크 사이즈 만큼 정확히 끊어서 사용하지 않고 오버랩 사이즈만큼 땡겨서 앞의 결과와 겹치게 함)
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def section_chunking(text: str) -> List[str]:
    # 특정 패턴 기준으로 텍스트 분리 -> 줄바꿈 -> #, ##, ### 다 인식 -> \s 헤더 뒤는 공백
    sections = re.split(r'\n#{1,3}\s+', text)
    # 헤더가 없는 일반 txt 파일이면 빈줄 기준으로 분리
    if len(sections) <= 1:
        sections = re.split(r'\n\s*\n', text)
    # 각 섹션 앞뒤 공백, 빈 섹션 제거
    return [s.strip() for s in sections if s.strip()]

# client -> Claude API 클라이언트 객체를 받음. 다른 청킹 함수들과 달리 API 호출 필요
def llm_chunking(text: str, client, model: str = "claude-opus-4-5") -> List[str]:
    # Claude에게 시키는 프롬프트(클로드가 문맥을 이해하고 의미 단위로 판단해서 자르도록)
    prompt = f"""다음 텍스트를 의미 단위로 나눠주세요.
각 청크는 <chunk>태그로 감싸주세요.

텍스트: 
{text}
"""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    # 클로드 응답에서 <chunk> ~ <\chunk> 사이 텍스트만 추출. 줄 바꿈은 .으로 인식
    raw = response.content[0].text
    chunks = re.findall(r'<chunk>(.*?)</chunk>', raw, re.DOTALL)
    return [c.strip() for c in chunks if c.strip()]

# 리모컨 같이 4가지 청킹 함수를 직접 부르는 대신 chunk_text 하나로 통일해서 사용하는 것
def chunk_text(text: str, method: str = "section", **kwargs) -> List[str]:
    if method == "fixed":
        return fixed_chunking(text, **kwargs)
    elif method == "overlap":
        return overlap_chunking(text, **kwargs)
    elif method == "section":
        return section_chunking(text, **kwargs)
    elif method == "llm":
        return llm_chunking(text, **kwargs)
    else:
        raise ValueError(f"알 수 없는 방법: {method}")
    
    # 여기서 **kwargs란? -> 각 청킹 함수마다 필요한 파라미터가 다름
    # 그래서 이게 파라미터들을 그대로 받아서 전달하는 역할