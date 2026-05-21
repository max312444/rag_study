import re
from typing import List

def fixed_chunking(text: str, chunk_size: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return [c for c in chunks if c.strip()]

def overlap_chunking(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def section_chunking(text: str) -> List[str]:
    sections = re.split(r'\n#{1, 3}\s+', text)
    if len(sections) <= 1:
        sections = re.split(r'\n\s*\n', text)
    return [s.strip() for s in sections if s.strip()]

def llm_chunking(text: str, client, model: str = "claude-opus-4-5") -> List[str]:
    prompt = f"""다음 텍스트를 의미 단위로 나눠주세요.
각 청크는 <chunk>태그로 감싸주세요.

텍스트: 
{text}
"""
    response = client.message.create(
        model=model,
        max_tokens=4096,
        message=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text
    chunks = re.findall(r'<chunk>(.*?)</chunk>', raw, re.DOTALL)
    return [c.strip() for c in chunks if c.strip()]

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