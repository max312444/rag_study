import asyncio
# MCP 서버 객체
from mcp.server import Server
# Claude Desktop이랑 연결하는 통신 방식
from mcp.server.stdio import stdio_server
# MCP에서 Tool 정의할 때 쓰는 타입
from mcp.types import Tool, TextContent
from retriever import retrieve, compare_methods
from indexer import index_file, list_indexed_docs
# 외부 tool 연결
from datetime import datetime
from duckduckgo_search import DDGS

# @server.list_tools() → Claude가 "어떤 도구 있어?" 물어볼 때 응답
# Tool마다 name, description, inputSchema 정의
# description 이 핵심! → Claude가 이걸 읽고 어떤 Tool 쓸지 판단
server = Server("rag-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_docs",
            description="문서에서 질문과 관련된 내용을 검색합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 질문"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 청크 수 (기본값 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="compare_chunking",
            description="같은 질문을 청킹 방법별로 검색해서 결과를 비교합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "비교할 질문"
                    }
                },
                "required": ["query"]
            }
        ),
                Tool(
            name="get_current_time",
            description="현재 날짜와 시간을 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="web_search",
            description="인터넷에서 최신 정보를 검색합니다. 문서에 없는 최신 정보가 필요할 때 사용합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 내용"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculate",
            description="수학 계산을 합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수식 (예: 2 + 3 * 4)"
                    }
                },
                "required": ["expression"]
            }
        ),
    ]

# @server.call_tool() → Claude가 Tool 실제로 호출할 때 실행
# name → 어떤 Tool인지 (search_docs or compare_chunking)
# arguments → Claude가 채워서 보낸 파라미터
# arguments.get("top_k", 3) → top_k 없으면 기본값 3 사용
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    
    if name == "search_docs":
        query = arguments["query"]
        top_k = arguments.get("top_k", 3)
        
        results = retrieve(query, top_k=top_k)
        
        if not results:
            text = "검색 결과가 없습니다."
        else:
            text = f"질문: {query}\n\n"
            for r in results:
                text += f"#{r['rank']} | 유사도: {r['score']:.4f} | {r['method']}\n"
                text += f"{r['text']}\n\n"
        
        return [TextContent(type="text", text=text)]

    elif name == "compare_chunking":
        query = arguments["query"]
        results = compare_methods(query, top_k=2)
        
        text = f"질문: {query}\n\n"
        for method, chunks in results.items():
            text += f"[{method}]\n"
            for r in chunks:
                text += f"  유사도: {r['score']:.4f} | {r['text'][:100]}...\n"
            text += "\n"
        
        return [TextContent(type="text", text=text)]
    
    elif name == "get_current_time":
        now = datetime.now()
        text = f"현재 시간: {now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}"
        return [TextContent(type="text", text=text)]

    elif name == "web_search":
        query = arguments["query"]
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"제목: {r['title']}\n내용: {r['body']}\n링크: {r['href']}")
        
        text = f"검색어: {query}\n\n" + "\n\n".join(results)
        return [TextContent(type="text", text=text)]

    elif name == "calculate":
        expression = arguments["expression"]
        result = eval(expression)
        text = f"{expression} = {result}"
        return [TextContent(type="text", text=text)]
    
    else:
        return [TextContent(type="text", text=f"알 수 없는 Tool: {name}")]
    
# stdio_server() → Claude Desktop이랑 표준입출력으로 통신
# server.run() → MCP 서버 시작, Claude 연결 대기
# asyncio.run(main()) → 비동기 서버 실행
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())