#from models.BochaSearchAgent import SearchAgent
from models.SearchMCPAgent import SearchAgent

if __name__ == '__main__':
    # 同步查询示例
    # result = SearchAgent.query("为什么选择langchain？")
    # print("同步查询结果:", result)
    
    # 流式查询示例
    print("\n流式查询结果:")
    for chunk in SearchAgent.stream_query("2025年7月甘肃省新能源发电的限电情况"):
        print(chunk, end="", flush=True)
    print()
    
    # 异步查询示例
    # import asyncio
    
    # async def test_async_query():
    #     result = await SearchAgent.aquery("为什么选择langchain？")
    #     print("\n异步查询结果:", result)
        
    #     print("\n异步流式查询结果:")
    #     async for chunk in SearchAgent.astream_query("为什么选择langchain？"):
    #         print(chunk, end="", flush=True)
    #     print()
    
    # # 运行异步测试
    # asyncio.run(test_async_query())

# from models.bocha_mcp_tools import BochaSearcher
# import asyncio
# from mcp.server.fastmcp import Context

# class MockContext(Context):
#     async def info(self, message: str):
#         print(f"[INFO] {message}")
    
#     async def error(self, message: str):
#         print(f"[ERROR] {message}")
    
#     async def debug(self, message: str):
#         print(f"[DEBUG] {message}")

# async def main():
#     searcher = BochaSearcher()
#     result = await searcher.bocha_ai_search("为什么选择langchain？", ctx=MockContext())
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())