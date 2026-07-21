'''Add SearchToolAgent.py
Author: liyu
Date: 2025-11-10 19:20
Add at: 2025-11-10 19:20
Basic search tool agent via duckduckgo search
    1. Add SearchToolAgent.py, which is used to wrap the search tool for the project.
    2. Use DuckDuckGoSearchResults to search the internet.
    3. Use ChatOpenAI to generate the answer.
'''

from typing import *
from langchain_core.tools import Tool, BaseTool
from langchain_core.prompts import StringPromptTemplate
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI
import os
import re
import concurrent.futures
import asyncio

api_key = os.getenv("OpenAI_API_KEY")
llm = ChatOpenAI(model="Qwen3-235B-A22B-Instruct-2507", base_url="http://localhost:11434/v1", api_key=api_key,
                 temperature=0.0)
agent_template = """
你现在是一个{role}。这里是一些已知信息：
{background_infomation}
{agent_scratchpad}
{question_guide}：{input}
{answer_format}
"""

class CustomPromptTemplate(StringPromptTemplate):
    '''
    Custom prompt template for the search tool agent.
    '''
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        # 设置agent_scratchpad为空字符串，避免缺失变量错误
        kwargs["agent_scratchpad"] = ""
        
        # 没有互联网查询信息
        if len(intermediate_steps) == 0:
            background_infomation = "\n"
            role = "傻瓜机器人"
            question_guide = "我现在有一个问题"
            answer_format = "如果你知道答案，请直接给出你的回答！如果你不知道答案，请你只回答\"DeepSearch('搜索词')\"，并将'搜索词'替换为你认为需要搜索的关键词，除此之外不要回答其他任何内容。\n\n下面请回答我上面提出的问题！"

        # 返回了背景信息
        else:
            # 根据 intermediate_steps 中的 AgentAction 拼装 background_infomation
            background_infomation = "\n\n你还有这些已知信息作为参考：\n\n"
            action, observation = intermediate_steps[0]
            background_infomation += f"{observation}\n"
            role = "聪明的 AI 助手"
            question_guide = "请根据这些已知信息回答我的问题"
            answer_format = ""

        kwargs["background_infomation"] = background_infomation
        kwargs["role"] = role
        kwargs["question_guide"] = question_guide
        kwargs["answer_format"] = answer_format
        return self.template.format(**kwargs)
    
class CustomSearchTool(BaseTool):
    name: str = "DeepSearch"
    description: str = ""

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        return DuckDuckGoSearchTool.search(query = query)

    async def _arun(self, query: str):
        raise DuckDuckGoSearchTool.asearch(query = query)


# 创建线程池用于并发搜索
search_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=5,  # 设置适当的线程数，避免过多请求
    thread_name_prefix="SearchAgent"
)

class DuckDuckGoSearchTool:
    '''
    DuckDuckGo search tool.
    Supported sync and async search with multi-threading.
    '''
    
    @staticmethod
    def _perform_search(query: str, region: str = "zh-cn", max_results: int = 30) -> str:
        """
        执行实际搜索的内部方法，可在线程池中运行
        """
        try:
            api_wrapper = DuckDuckGoSearchAPIWrapper(region=region, max_results=max_results)
            search_tool = DuckDuckGoSearchResults(api_wrapper=api_wrapper)

            result = ""
            for chunk in search_tool.stream(query):
                result += chunk
            return result
        except Exception as e:
            print(f"搜索错误: {str(e)}")
            return f"搜索过程中发生错误: {str(e)}"
    
    def search(self, query: str = ""):
        query = query.strip()
        if query == "":
            return ""
        
        # 使用线程池执行搜索，避免阻塞主线程
        future = search_thread_pool.submit(self._perform_search, query)
        try:
            # 设置超时时间，避免单个搜索任务无限阻塞
            result = future.result(timeout=30)  # 30秒超时
            return result
        except concurrent.futures.TimeoutError:
            return "搜索超时，请稍后重试"
        except Exception as e:
            return f"搜索执行错误: {str(e)}"
    
    async def asearch(self, query: str = ""):
        query = query.strip()
        if query == "":
            return ""
        
        # 在异步环境中使用线程池执行同步搜索操作
        loop = asyncio.get_event_loop()
        try:
            # 使用run_in_executor在后台线程中执行搜索
            result = await loop.run_in_executor(
                search_thread_pool, 
                self._perform_search, 
                query
            )
            return result
        except Exception as e:
            return f"异步搜索错误: {str(e)}"
    
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # group1 = 调用函数名字
        # group2 = 传入参数
        match = re.match(r'^[\s\w]*(DeepSearch)\(([^\)]+)\)', llm_output, re.DOTALL)

        # 如果 llm 没有返回 DeepSearch() 则认为直接结束指令
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        # 否则的话都认为需要调用 Tool
        else:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class SearchAgent:
    '''
    Search agent.
    Supported sync and async search.
    '''
    def query(query: str = ""):
        # 创建搜索工具，确保有明确的描述
        tools = [
                    Tool.from_function(
                        func=DuckDuckGoSearchTool.search,
                        name="DeepSearch",
                        description="使用DuckDuckGo搜索引擎搜索信息，输入搜索查询词，返回搜索结果。"
                    )
                ]
        
        # 使用最新的LangChain API创建带有工具调用功能的代理
        # 1. 首先创建一个简单的提示模板，不包含agent_scratchpad
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。请回答以下问题，如果需要更多信息，请使用DeepSearch工具。"),
            ("human", "{input}")
        ])
        
        # 2. 使用RunnablePassthrough来创建一个正确格式的输入
        
        # 3. 创建工具调用链
        # 注意：在较新的LangChain版本中，我们需要确保提示模板的格式正确
        chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: ""
        ) | prompt | llm.bind_tools(tools)
        
        # 4. 创建一个简单的执行器来处理工具调用
        # 手动执行搜索逻辑，避免使用已弃用的AgentExecutor方式
        result = chain.invoke({"input": query})
        
        # 5. 检查是否需要调用工具
        if hasattr(result, 'tool_calls') and result.tool_calls:
            # 执行工具调用
            tool_call = result.tool_calls[0]
            
            if tool_call['name'] == "DeepSearch":
                # 获取工具调用参数
                args = tool_call['args']
                # 从args中获取搜索查询（支持'__arg1'或直接使用第一个值）
                search_query = args.get('__arg1') or list(args.values())[0] if args else query
                
                # 执行搜索
                tool_result = DuckDuckGoSearchTool.search(search_query)
                
                # 使用搜索结果生成最终回答
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个智能助手。基于以下信息回答问题。"),
                    ("human", "问题: {input}\n\n信息: {tool_result}")
                ])
                final_chain = final_prompt | llm
                final_result = final_chain.invoke({"input": query, "tool_result": tool_result})
                return {"output": final_result.content}
        
        # 如果没有工具调用，直接返回模型的回答
        return {"output": result.content}
    
    def stream_query(query: str = ""):
        """
        同步流式输出方法
        返回一个生成器，可以逐步获取回答内容
        """
        # 创建搜索工具
        tools = [
                    Tool.from_function(
                        func=DuckDuckGoSearchTool.search,
                        name="DeepSearch",
                        description="使用DuckDuckGo搜索引擎搜索信息，输入搜索查询词，返回搜索结果。"
                    )
                ]
        
        # 创建提示模板
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。请回答以下问题，如果需要更多信息，请使用DeepSearch工具。"),
            ("human", "{input}")
        ])
        
        # 创建工具调用链
        
        chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: ""
        ) | prompt | llm.bind_tools(tools)
        
        # 执行链获取结果
        result = chain.invoke({"input": query})
        
        # 检查是否需要调用工具
        if hasattr(result, 'tool_calls') and result.tool_calls:
            tool_call = result.tool_calls[0]
            
            if tool_call['name'] == "DeepSearch":
                # 获取搜索查询
                args = tool_call['args']
                search_query = args.get('__arg1') or list(args.values())[0] if args else query
                
                # 执行搜索
                tool_result = DuckDuckGoSearchTool.search(search_query)
                
                # 使用搜索结果生成最终回答（流式）
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个智能助手。基于以下信息回答问题。"),
                    ("human", "问题: {input}\n\n信息: {tool_result}")
                ])
                final_chain = final_prompt | llm
                
                # 使用stream方法进行流式输出
                for chunk in final_chain.stream({"input": query, "tool_result": tool_result}):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
        else:
            # 如果没有工具调用，直接流式输出模型的回答
            for chunk in llm.stream(prompt.format_messages(input=query)):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
    
    async def aquery(query: str = ""):
        tools = [
                    Tool.from_function(
                        func=DuckDuckGoSearchTool.asearch,  # 使用异步版本的search函数
                        name="DeepSearch",
                        description="使用DuckDuckGo搜索引擎搜索信息，输入搜索查询词，返回搜索结果。"
                    )
                ]
        
        # 使用最新的LangChain API创建带有工具调用功能的代理
        # 1. 首先创建一个简单的提示模板，不包含agent_scratchpad
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。请回答以下问题，如果需要更多信息，请使用DeepSearch工具。"),
            ("human", "{input}")
        ])
        
        # 2. 使用RunnablePassthrough来创建一个正确格式的输入
        
        # 3. 创建工具调用链
        # 注意：在较新的LangChain版本中，我们需要确保提示模板的格式正确
        chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: ""
        ) | prompt | llm.bind_tools(tools)
        
        # 4. 创建一个简单的执行器来处理工具调用
        # 手动执行搜索逻辑，避免使用已弃用的AgentExecutor方式
        # 重要：使用await等待异步操作完成
        result = await chain.ainvoke({"input": query})
        
        # 5. 检查是否需要调用工具
        if hasattr(result, 'tool_calls') and result.tool_calls:
            # 执行工具调用
            tool_call = result.tool_calls[0]
            
            if tool_call['name'] == "DeepSearch":
                # 获取工具调用参数
                args = tool_call['args']
                # 从args中获取搜索查询（支持'__arg1'或直接使用第一个值）
                search_query = args.get('__arg1') or list(args.values())[0] if args else query
                
                # 执行搜索 - 使用await等待异步搜索完成
                tool_result = await DuckDuckGoSearchTool.asearch(search_query)
                
                # 使用搜索结果生成最终回答
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个智能助手。基于以下信息回答问题。"),
                    ("human", "问题: {input}\n\n信息: {tool_result}")
                ])
                final_chain = final_prompt | llm
                # 使用await等待最终回答生成
                final_result = await final_chain.ainvoke({"input": query, "tool_result": tool_result})
                return {"output": final_result.content}
        
        # 如果没有工具调用，直接返回模型的回答
        return {"output": result.content}
    
    async def astream_query(query: str = ""):
        """
        异步流式输出方法
        返回一个异步生成器，可以逐步获取回答内容
        """
        # 创建搜索工具
        tools = [
                    Tool.from_function(
                        func=DuckDuckGoSearchTool.asearch,
                        name="DeepSearch",
                        description="使用DuckDuckGo搜索引擎搜索信息，输入搜索查询词，返回搜索结果。"
                    )
                ]
        
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。请回答以下问题，如果需要更多信息，请使用DeepSearch工具。"),
            ("human", "{input}")
        ])
        
        # 创建工具调用链
        chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: ""
        ) | prompt | llm.bind_tools(tools)
        
        # 执行链获取结果（异步）
        result = await chain.ainvoke({"input": query})
        
        # 检查是否需要调用工具
        if hasattr(result, 'tool_calls') and result.tool_calls:
            tool_call = result.tool_calls[0]
            
            if tool_call['name'] == "DeepSearch":
                # 获取搜索查询
                args = tool_call['args']
                search_query = args.get('__arg1') or list(args.values())[0] if args else query
                
                # 异步执行搜索
                tool_result = await DuckDuckGoSearchTool.asearch(search_query)
                
                # 使用搜索结果生成最终回答（流式异步）
                final_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个智能助手。基于以下信息回答问题。"),
                    ("human", "问题: {input}\n\n信息: {tool_result}")
                ])
                final_chain = final_prompt | llm
                
                # 使用astream方法进行异步流式输出
                async for chunk in final_chain.astream({"input": query, "tool_result": tool_result}):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
        else:
            # 如果没有工具调用，直接异步流式输出模型的回答
            async for chunk in llm.astream(prompt.format_messages(input=query)):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

if __name__ == '__main__':
    # 同步查询示例
    result = SearchAgent.query("llamaindex 介绍")
    print("同步查询结果:", result)
    
    # 流式查询示例
    print("\n流式查询结果:")
    for chunk in SearchAgent.stream_query("llamaindex 介绍"):
        print(chunk, end="", flush=True)
    print()
    
    # 异步查询示例
    import asyncio
    
    async def test_async_query():
        result = await SearchAgent.aquery("llamaindex 与 langchain 比较")
        print("\n异步查询结果:", result)
        
        print("\n异步流式查询结果:")
        async for chunk in SearchAgent.astream_query("llamaindex 与 langchain 比较"):
            print(chunk, end="", flush=True)
        print()
    
    # 运行异步测试
    asyncio.run(test_async_query())
