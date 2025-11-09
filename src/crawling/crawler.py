import re
import asyncio
import aiohttp
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

# ---------------------------
# Async Crawling Functions
# ---------------------------

async def fetch_url_async(session: aiohttp.ClientSession, url: str, timeout: int = 15) -> Optional[str]:
    """Async fetch a single URL"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout), ssl=False) as response:
            if response.status == 200:
                return await response.text()
    except Exception:
        return None
    return None


async def crawl_all(urls: List[str]) -> List[Dict[str, Optional[str]]]:
    """Crawl multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Pair URLs with their results
        crawled = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception) or result is None:
                crawled.append({"url": url, "html": None})
            else:
                crawled.append({"url": url, "html": result})
        
        return crawled


# Keep sync version as fallback
def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.text
    except Exception:
        return None
    return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
