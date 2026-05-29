import os
import anthropic
from dotenv import load_dotenv
from retriever import retrieve
from hybrid_retriever import hybrid_search
from reranker import rerank

load_dotenv()

# LLM이 사용할 수 있는 도구 정의
# LLM이 스스로 어떤 도구를 쓸지, 언제 쓸지, 몇 번 쓸지 판단
TOOLS = [
    {
        "name": "search_docs",
        "description": "문서에서 질문과 관련된 내용을 벡터 검색합니다. 간단한 질문에 적합합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 질문"},
                "top_k": {"type": "integer", "description": "변환할 청크 수", "default": 3}
            },
            "required": ["query"]
        }
    },
    {
        "name": "advanced_search",
        "description": "하이브리드 검색과 리랭킹을 사용해 고품질 검색을 합니다. 복잡하거나 중요한 질문에 적합합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 질문"},
                "top_k": {"type": "integer", "description": "반환할 청크 수", "default": 3}
            },
            "required": ["query"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    if name == "search_docs":
        results = retrieve(inputs["query"], top_k=inputs.get("top_k", 3))
        if not results:
            return "검색 결과 없음"
        return "\n\n".join(f"[유사도 {r['score']:.4f}]\n{r['text']}" for r in results)
    
    elif name == "advanced_search":
        candidates = hybrid_search(inputs["query"], top_k=inputs.get("top_k", 3) * 3)
        results = rerank(inputs["query"], candidates, top_k=inputs.get("top_k", 3))
        if not results:
            return "검색 결과 없음"
        return "\n\n".join(f"[관련도 {r['rerank_score']:.4f}]\n{r['text']}" for r in results)
    
    return "알 수 없는 도구"


def agentic_ask(query: str, api_key: str = None, max_iterations: int = 3):
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("API 키 없음 - Agentic RAG는 LLM이 필수입니다.")
        return
    
    client = anthropic.Anthropic(api_key=key)

    print(f"\n=== [Agentic RAG] 질문: {query} ===")

    messages = [{"role": "user", "content": query}]

    for iteration in range(max_iterations):
        print(f"\n[반복 {iteration + 1} / {max_iterations}]")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )

        print(f"  stop_reason: {response.stop_reason}")

        # LLM이 도구 호출 없이 답변 -> 충분한 정보 모였다고 판단한 것
        if response.stop_reason == "end_turn":
            answer = next(b.text for b in response.content if hasattr(b, "text"))
            print(f"\n[최종 답변]\n{answer}")
            return
        
        # LLM이 도구를 호출 -> 정보가 더 필요하다고 판단한 것
        if response.stop_reason == "tool_use":
            # assistant 응답을 대화 히스토리에 추가
            messages.append({"role": "assistant", "content": response.content})

            tool_result = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  도구 호출: {block.name}({block.input})")
                    result = execute_tool(block.name, block.input)
                    print(f"  결과 미리보기: {result[:100]}...")
                    tool_result.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # 도구 결과를 대화 히스토리에 추가 -> 다음 반복해서 LLM이 참고
            messages.append({"role": "user", "content": tool_result})

    print("\n[최대 반복 횟수 도달]")

if __name__ == "__main__":
    agentic_ask("RAG란 무엇이고 어떻게 동작하나요?", api_key=os.getenv("ANTHROPIC_API_KEY"))