import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
from dotenv import load_dotenv

from .consts import (
    CACHE_DIR,
    CACHE_FILE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MODEL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
    OPENAI_API_URL,
)


class OpenAIClient:
    """Client for making asynchronous requests to the OpenAI API with caching capability."""

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True):
        # Load API key from parameter or .env file
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Provide it as parameter or set in .env file"
            )

        self.base_url = OPENAI_API_URL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Initialize cache
        self.cache_enabled = cache_enabled
        if cache_enabled:
            cache_dir = Path(CACHE_DIR)
            cache_dir.mkdir(exist_ok=True)
            self.cache_file = cache_dir / CACHE_FILE
            self.cache = self._load_cache()
        else:
            self.cache = {}

        # Session management
        self._session = None

        # Configuration
        self.session_timeout = aiohttp.ClientTimeout(
            total=DEFAULT_TIMEOUT, connect=DEFAULT_CONNECT_TIMEOUT
        )
        self.retry_attempts = DEFAULT_RETRY_ATTEMPTS
        self.retry_delay = DEFAULT_RETRY_DELAY

    def _load_cache(self) -> Dict[str, Any]:
        """Load the cache from the cache file or create a new one."""
        if not self.cache_enabled:
            return {}

        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Cache file corrupted or unreadable: {e}. Creating new cache.")
                return {}
        return {}

    async def _save_cache(self) -> None:
        """Save the current cache to the cache file."""
        if not self.cache_enabled:
            return

        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Failed to save cache: {e}")

    def _generate_cache_key(
        self, messages: List[Dict[str, Any]], model: str, **kwargs
    ) -> str:
        """Generate a unique key for the cache based on request parameters."""
        # Create a string representation of the request parameters
        # Remove non-deterministic parameters from cache key
        cache_params = {
            "messages": messages,
            "model": model,
            **{k: v for k, v in kwargs.items() if k not in ["stream", "user"]},
        }
        param_str = json.dumps(cache_params, sort_keys=True, ensure_ascii=False)

        # Generate a hash of the parameters
        return hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp ClientSession with optimized settings."""
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(
                limit=100, ttl_dns_cache=300, use_dns_cache=True, keepalive_timeout=30
            )
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.session_timeout,
                connector=conn,
                raise_for_status=False,  # We'll handle status codes manually
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session and save cache."""
        if self.cache_enabled:
            await self._save_cache()

        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def chat_completion(
        self, messages: List[Dict[str, Any]], model: str = DEFAULT_MODEL, **kwargs
    ) -> Dict[str, Any]:
        """
        Create an asynchronous chat completion request to OpenAI API with caching.

        Args:
            messages: List of message objects with 'role' and 'content'
            model: OpenAI model to use (default from consts)
            **kwargs: Additional parameters to pass to the API

        Returns:
            The API response (from cache or fresh request)

        Raises:
            ValueError: If messages are invalid
            aiohttp.ClientResponseError: If API request fails
            asyncio.TimeoutError: If request times out
        """
        # Validate inputs
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        # Check if high temperature is set (requesting randomness)
        temperature = kwargs.get("temperature", 0)
        bypass_cache = temperature >= 0.7 or not self.cache_enabled

        cache_key = None
        if not bypass_cache:
            # Generate a cache key for this request
            cache_key = self._generate_cache_key(messages, model, **kwargs)

            # Check if response is in cache
            if cache_key in self.cache:
                print("Using cached response")
                return self.cache[cache_key]

        # Make the API request with retries
        url = f"{self.base_url}/chat/completions"

        payload = {"model": model, "messages": messages, **kwargs}

        # Log approximate payload size for debugging
        payload_size = len(json.dumps(payload).encode("utf-8")) / 1024
        print(f"Payload size: {payload_size:.2f} KB")

        session = await self.get_session()
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                async with session.post(url, json=payload) as response:
                    # Handle different HTTP status codes
                    if response.status == 200:
                        result = await response.json()

                        # Cache the response if needed
                        if cache_key and self.cache_enabled:
                            self.cache[cache_key] = result
                            await self._save_cache()

                        return result

                    elif response.status == 429:  # Rate limit
                        if attempt < self.retry_attempts - 1:
                            # Try to get retry-after header
                            retry_after = response.headers.get("retry-after")
                            wait_time = (
                                int(retry_after)
                                if retry_after
                                else self.retry_delay * (2**attempt)
                            )
                            print(f"Rate limited. Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Rate limit exceeded: {error_text}",
                            )

                    elif response.status >= 400:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_message = error_data.get("error", {}).get(
                                "message", error_text
                            )
                        except json.JSONDecodeError:
                            error_message = error_text

                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"API Error: {error_message}",
                        )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    print(
                        f"Connection error (attempt {attempt + 1}/{self.retry_attempts}): {str(e)}"
                    )
                    print(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"All retry attempts failed. Last error: {str(e)}")
                    raise

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unexpected error: no response received")


# Helper functions for batch processing
async def process_batch(
    client: OpenAIClient, batch_requests: List[Dict[str, Any]], max_concurrent: int = 10
) -> List[Union[Dict[str, Any], Exception]]:
    """
    Process a batch of requests in parallel with concurrency control.

    Args:
        client: The OpenAI client instance
        batch_requests: List of dictionaries with args for chat_completion
        max_concurrent: Maximum number of concurrent requests

    Returns:
        Results from all requests (successful responses or exceptions)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_request(request: Dict[str, Any]):
        async with semaphore:
            messages = request.get("messages", [])
            model = request.get("model", DEFAULT_MODEL)
            kwargs = {
                k: v for k, v in request.items() if k not in ["messages", "model"]
            }
            return await client.chat_completion(messages, model, **kwargs)

    tasks = [process_single_request(request) for request in batch_requests]
    return await asyncio.gather(*tasks, return_exceptions=True)


# Example usage function
async def example_usage():
    """
    Example of how to use the OpenAI client.
    """
    async with OpenAIClient() as client:
        # Single request
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        try:
            response = await client.chat_completion(messages)
            print("Response:", response["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Error: {e}")

        # Batch requests
        batch_requests = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "model": "gpt-4o",
                "temperature": 0.1,
            },
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "model": "gpt-4o",
                "temperature": 0.1,
            },
        ]

        try:
            batch_results = await process_batch(client, batch_requests)
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"Request {i} failed: {result}")
                else:
                    print(
                        f"Request {i} result: {result['choices'][0]['message']['content']}"
                    )
        except Exception as e:
            print(f"Batch processing error: {e}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
