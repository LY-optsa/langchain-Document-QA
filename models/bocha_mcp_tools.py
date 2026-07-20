from mcp.server.fastmcp import FastMCP, Context
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import urllib.parse
import sys
import traceback
import asyncio
import json
from datetime import datetime, timedelta
import time
import re
import os


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)

class BochaSearcher:
    boch_api_key = os.environ.get("BOCHA_API_KEY", "")
    if not boch_api_key:
        raise ValueError (
            "Error: Bocha API key is not configured. Please set the "
            "BOCHA_API_KEY environment variable."
        )
    
    BOCHA_WEB_SEARCH_URL = "https://api.bochaai.com/v1/web-search?utm_source=bocha-mcp-local"
    BOCHA_AI_SEARCH_URL = "https://api.bochaai.com/v1/ai-search?utm_source=bocha-mcp-local"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style that's easier for LLMs to process"""
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)
    
    async def bocha_web_search(self, query: str, ctx: Context = None, freshness: str = "noLimit", count: int = 10) -> str:
        try:
            payload = {
                "query": query,
                "summary": True,
                "freshness": freshness,
                "count": count
            }

            headers = {
                "Authorization": f"Bearer {self.boch_api_key}",
                "Content-Type": "application/json",
            }

            await ctx.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BOCHA_WEB_SEARCH_URL, headers=headers, json=payload, timeout=10.0
                )

                response.raise_for_status()
                resp = response.json()
                if "data" not in resp:
                    return "Search error."
                
                data = resp["data"]

                if "webPages" not in data:
                    return "No results found."

                results = []
                for result in data["webPages"]["value"]:
                    results.append(
                        f"Title: {result['name']}\n"
                        f"URL: {result['url']}\n"
                        f"Description: {result['summary']}\n"
                        f"Published date: {result['datePublished']}\n"
                        f"Site name: {result['siteName']}"
                    )

                return "\n\n".join(results)

        except httpx.HTTPStatusError as e:
            return f"Bocha Web Search API HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Error communicating with Bocha Web Search API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
    
    async def bocha_ai_search(self, query: str, ctx: Context = None, freshness: str = "noLimit", count: int = 10):
        try:
            payload = {
                "query": query,
                "freshness": freshness,
                "count": count,
                "answer": False,
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {self.boch_api_key}",
                "Content-Type": "application/json",
            }

            await ctx.info(f"Searching Bocha AI for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BOCHA_AI_SEARCH_URL, headers=headers, json=payload, timeout=10.0
                )

                response.raise_for_status()
                response = response.json()
                results = []
                if "messages" in response:
                    for message in response["messages"]:
                        content = {}
                        try:
                            content = json.loads(message["content"])
                        except:
                            content = {}
                            
                        # 网页
                        if message["content_type"] == "webpage":
                            if "value" in content:
                                for item in content["value"]:
                                    results.append(
                                        f"Title: {item['name']}\n"
                                        f"URL: {item['url']}\n"
                                        f"Description: {item['summary']}\n"
                                        f"Published date: {item['datePublished']}\n"
                                        f"Site name: {item['siteName']}"
                                    )
                        elif message["content_type"] != "image" and message["content"] != "{}":
                            results.append(message["content"])

                if not results:
                    return "No results found."
                
                return "\n\n".join(results)

        except httpx.HTTPStatusError as e:
            return f"Bocha AI Search API HTTP error occurred: {e.response.status_code} - {e.response.text}"
        except httpx.RequestError as e:
            return f"Error communicating with Bocha AI Search API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
        
class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str, ctx: Context) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            await ctx.info(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"

            await ctx.info(
                f"Successfully fetched and parsed content ({len(text)} characters)"
            )
            return text

        except httpx.TimeoutException:
            await ctx.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPError as e:
            await ctx.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            await ctx.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"
        

server = FastMCP(
    "bocha-search-mcp",
    instructions="""
# Bocha Search MCP Server
                 
Bocha is a Chinese search engine for AI, This server provides tools for searching the web using Bocha Search API.
It allows you to get enhanced search details from billions of web documents, including weather, news, wikis, healthcare, train tickets, images, and more.

## Available Tools
                 
### 1. bocha_web_search 
Search with Bocha Web Search and get enhanced search details from billions of web documents, including page titles, urls, summaries, site names, site icons, publication dates, image links, and more.

### 2. bocha_ai_search
Search with Bocha AI Search, recognizes the semantics of search terms and additionally returns structured modal cards with content from vertical domains.

## Output Format

All search results will be formatted as text with clear sections for each
result item, including:

- Bocha Web search: Title, URL, Description, Published date and Site name
- Bocha AI search: Title, URL, Description, Published date, Site name, and structured data card

If the API key is missing or invalid, appropriate error messages will be returned.
"""
)

searcher = BochaSearcher()
fetcher = WebContentFetcher()

@server.tool()
async def web_search(query: str, ctx: Context, freshness: str = "noLimit", max_results: int = 20) -> str:
    """
    Search Bocha Web Search and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.bocha_web_search(query, ctx, freshness, max_results)
        return results
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"
    
@server.tool()
async def ai_search(query: str, ctx: Context, freshness: str = "noLimit", max_results: int = 20) -> str:
    """
    Search Bocha AI Search and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.bocha_ai_search(query, ctx, freshness, max_results)
        return results
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"
    
@server.tool()
async def fetch_content(url: str, ctx: Context) -> str:
    """
    Fetch and parse content from a webpage URL.

    Args:
        url: The URL of the webpage to fetch
        ctx: MCP context for logging
    """
    try:
        content = await fetcher.fetch_and_parse(url, ctx)
        return content
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while fetching content: {str(e)}"


def run_mcp():
    print("Running Bocha Search MCP Server...")
    server.run()

if __name__ == "__main__":
    run_mcp()