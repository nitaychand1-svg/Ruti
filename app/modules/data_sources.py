import asyncio
from functools import lru_cache

@lru_cache(maxsize=100)
async def async_fetch_news(ticker):
    await asyncio.sleep(0.05)
    return [f"News for {ticker}: Market up 2%."]
