"""
Rate limiting utilities for external API calls
"""
import asyncio
import time
from typing import List

class RateLimiter:
    """
    Simple rate limiter for external API calls
    """
    def __init__(self, calls_per_minute: int = 10):
        """
        Initialize a rate limiter
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.call_times: List[float] = []
        
    async def wait_if_needed(self) -> None:
        """
        Wait if we've made too many calls recently.
        """
        current_time = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]
        
        # If we've made too many calls, wait
        if len(self.call_times) >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.call_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Add current call
        self.call_times.append(time.time())

# Initialize common rate limiters
from config.settings import DDG_CALLS_PER_MINUTE, TAVILY_CALLS_PER_MINUTE, TRANSLATION_CALLS_PER_MINUTE

ddg_rate_limiter = RateLimiter(calls_per_minute=DDG_CALLS_PER_MINUTE)
tavily_rate_limiter = RateLimiter(calls_per_minute=TAVILY_CALLS_PER_MINUTE)
translation_rate_limiter = RateLimiter(calls_per_minute=TRANSLATION_CALLS_PER_MINUTE)