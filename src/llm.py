import os
import re
import time
import random
from typing import Optional, List, Dict, Any
from loguru import logger
from openai import OpenAI
from groq import Groq

from src.config import settings

class UnifiedCompletions:
    def __init__(self, client: "UnifiedLLMClient"):
        self.client = client

    def create(self, *args, **kwargs) -> Any:
        if kwargs.get("stream", False):
            return self.client._create_completion_stream(*args, **kwargs)
        return self.client._create_completion(*args, **kwargs)

class UnifiedChat:
    def __init__(self, client: "UnifiedLLMClient"):
        self.completions = UnifiedCompletions(client)

class UnifiedLLMClient:
    """
    Multi-provider LLM client providing seamless fallback and load-sharing
    between Groq and Google Gemini API.
    
    If Groq hits daily token limits (TPD) or rate limits (429), it automatically
    and immediately fails over to Google Gemini without failing the user's request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: float = 20.0
    ):
        self.timeout = timeout
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        
        # Resolve keys
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        self.gemini_api_key = (
            settings.GEMINI_API_KEY
            or settings.GOOGLE_API_KEY
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        
        if api_key:
            if api_key.startswith("AIza"):
                self.gemini_api_key = api_key
            else:
                self.groq_api_key = api_key
                
        # Primary reference key for backward compatibility
        self.api_key = self.gemini_api_key if self.provider == "gemini" else (self.groq_api_key or self.gemini_api_key or "local-ollama")

        if not self.groq_api_key and not self.gemini_api_key and self.provider not in ["auto", "ollama"]:
            raise ValueError(
                "No LLM API key configured. Please set GROQ_API_KEY or GEMINI_API_KEY in your .env file."
            )

        # Initialize clients
        self._ollama_last_failure_time = 0.0
        self._gemini_last_exhausted_time = 0.0
        self.ollama_client: Optional[OpenAI] = None
        try:
            ollama_timeout = float(getattr(settings, "OLLAMA_TIMEOUT", 1.0))
            self.ollama_client = OpenAI(
                base_url=getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
                api_key="ollama",
                timeout=ollama_timeout
            )
        except Exception as e:
            logger.warning(f"Ollama client init error: {e}")

        self.groq_client: Optional[Groq] = None
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"Groq client init error: {e}")

        self.gemini_client: Optional[OpenAI] = None
        if self.gemini_api_key:
            try:
                self.gemini_client = OpenAI(
                    api_key=self.gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    timeout=min(self.timeout, 8.0)
                )
            except Exception as e:
                logger.warning(f"Gemini client init error: {e}")

        # Interface compatibility
        self.chat = UnifiedChat(self)

    def _create_completion(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Any:
        messages = messages or []
        
        # Determine priority based on provider setting and availability
        if self.provider == "gemini":
            providers = ["gemini", "groq", "ollama"]
        elif self.provider == "groq":
            providers = ["groq", "gemini", "ollama"]
        elif self.provider == "ollama":
            providers = ["ollama", "gemini", "groq"]
        else:  # "auto": Gemini primary, Groq fallback, Ollama ultimate safety net
            providers = ["gemini", "groq", "ollama"]

        last_error = None

        for p in providers:
            if p == "ollama" and self.ollama_client:
                if (time.time() - self._ollama_last_failure_time) < 3600.0 and self.provider != "ollama":
                    continue
                try:
                    ollama_model = getattr(settings, "OLLAMA_MODEL", "qwen2.5:3b")
                    call_kwargs = {
                        "model": ollama_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if response_format:
                        call_kwargs["response_format"] = response_format
                    call_kwargs.update(kwargs)
                    resp = self.ollama_client.chat.completions.create(**call_kwargs)
                    if resp and resp.choices and resp.choices[0].message.content:
                        return resp
                except Exception as e:
                    last_error = e
                    self._ollama_last_failure_time = time.time()
                    logger.warning(f"Ollama local call failed: {e}. Falling back to cloud...")

            elif p == "groq" and self.groq_client:
                # Prepare Groq candidate models (prioritizing high-speed 20b and 120b)
                groq_models = []
                if model and not model.startswith("gemini-") and model != "qwen/qwen3.8-27b":
                    groq_models.append(model)
                for gm in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                    if gm not in groq_models:
                        groq_models.append(gm)

                for m in groq_models:
                    for attempt in range(3):
                        try:
                            call_kwargs = {
                                "model": m,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            }
                            if response_format:
                                call_kwargs["response_format"] = response_format
                            call_kwargs.update(kwargs)
                            
                            resp = self.groq_client.chat.completions.create(**call_kwargs)
                            if resp and resp.choices:
                                content = resp.choices[0].message.content
                                if not content and getattr(resp.choices[0].message, "reasoning", None):
                                    reasoning_raw = resp.choices[0].message.reasoning
                                    m_ans = re.search(r'(?:So the answer is|Therefore the answer is|The answer is|Short Answer:|Thus, the answer is)\s*[:\*\s]*([^\n]+)', reasoning_raw, re.IGNORECASE)
                                    if m_ans:
                                        resp.choices[0].message.content = m_ans.group(1).strip().strip("'\"*")
                                    else:
                                        r_lines = [l.strip() for l in reasoning_raw.split('\n') if l.strip()]
                                        resp.choices[0].message.content = r_lines[-1] if r_lines else reasoning_raw
                                if resp.choices[0].message.content:
                                    return resp
                        except Exception as e:
                            last_error = e
                            err_str = str(e).lower()
                            if "tpd" in err_str or "tokens per day" in err_str or "daily limit" in err_str:
                                logger.warning(f"Groq daily quota limit reached ({m}). Attempting next model or provider...")
                                break
                            elif "429" in err_str or "rate limit" in err_str:
                                backoff = min(15, (2 ** attempt) + random.uniform(0.5, 1.5))
                                logger.warning(f"Groq 429 ({m}). Waiting {backoff:.1f}s...")
                                time.sleep(backoff)
                            else:
                                logger.warning(f"Groq call ({m}) error: {e}. Trying fallback...")
                                break

            elif p == "gemini" and self.gemini_client:
                # If Gemini recently failed with daily quota exhaustion, skip immediately
                if (time.time() - self._gemini_last_exhausted_time) < 3600.0:
                    continue

                # Prepare Gemini candidate models
                gemini_models = []
                if model and model.startswith("gemini-") and "pro" not in model and "2.5" not in model:
                    gemini_models.append(model)
                else:
                    gemini_models.append(settings.GEMINI_MODEL)
                if "gemini-3.8-flash" not in gemini_models:
                    gemini_models.append("gemini-3.8-flash")

                for m in gemini_models:
                    for attempt in range(3):
                        try:
                            call_kwargs = {
                                "model": m,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                            }
                            if response_format:
                                call_kwargs["response_format"] = response_format
                            call_kwargs.update(kwargs)

                            resp = self.gemini_client.chat.completions.create(**call_kwargs)
                            if resp and resp.choices and resp.choices[0].message.content:
                                return resp
                        except Exception as e:
                            last_error = e
                            err_str = str(e).lower()
                            # If Gemini complains about response_format json, retry without it
                            if "response_format" in err_str or "json_object" in err_str:
                                try:
                                    logger.info(f"Retrying Gemini {m} without explicit response_format...")
                                    clean_kwargs = dict(call_kwargs)
                                    clean_kwargs.pop("response_format", None)
                                    resp = self.gemini_client.chat.completions.create(**clean_kwargs)
                                    if resp and resp.choices and resp.choices[0].message.content:
                                        return resp
                                except Exception as e2:
                                    last_error = e2
                            
                            if "daily" in err_str or "per day" in err_str or "limit: 20" in err_str or "please retry in" in err_str:
                                self._gemini_last_exhausted_time = time.time()
                                logger.warning(f"Gemini daily quota exhausted ({m}). Switching immediately to secondary provider...")
                                break
                            elif "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str:
                                backoff = min(10, (2 ** attempt) + random.uniform(0.5, 1.5))
                                logger.warning(f"Gemini 429 ({m}). Waiting {backoff:.1f}s...")
                                time.sleep(backoff)
                            else:
                                logger.warning(f"Gemini call ({m}) error: {e}. Trying next model...")
                                break

        # If all providers and models failed
        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain completion from configured LLM providers.")

    def _create_completion_stream(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        **kwargs
    ):
        """Yields string tokens in real-time as they are streamed by the model."""
        messages = messages or []
        kwargs.pop("stream", None)
        kwargs.pop("response_format", None)

        if self.provider == "gemini":
            providers = ["gemini", "groq", "ollama"]
        elif self.provider == "groq":
            providers = ["groq", "gemini", "ollama"]
        elif self.provider == "ollama":
            providers = ["ollama", "gemini", "groq"]
        else:  # "auto": Gemini primary, Groq fallback, Ollama ultimate safety net
            providers = ["gemini", "groq", "ollama"]

        for p in providers:
            if p == "ollama" and self.ollama_client:
                if (time.time() - self._ollama_last_failure_time) < 30.0 and self.provider != "ollama":
                    continue
                try:
                    ollama_model = getattr(settings, "OLLAMA_MODEL", "qwen2.5:3b")
                    stream = self.ollama_client.chat.completions.create(
                        model=ollama_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **kwargs
                    )
                    for chunk in stream:
                        content = chunk.choices[0].delta.content if (chunk.choices and chunk.choices[0].delta) else ""
                        if content:
                            yield content
                    return
                except Exception as e:
                    self._ollama_last_failure_time = time.time()
                    logger.warning(f"Ollama stream error: {e}. Falling back to cloud...")

            elif p == "groq" and self.groq_client:
                try:
                    m = model or settings.LLM_MODEL
                    if m.startswith("gemini-"):
                        m = settings.LLM_MODEL
                    stream = self.groq_client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **kwargs
                    )
                    for chunk in stream:
                        content = chunk.choices[0].delta.content if (chunk.choices and chunk.choices[0].delta) else ""
                        if content:
                            yield content
                    return
                except Exception as e:
                    logger.warning(f"Groq stream error: {e}. Falling back to Gemini...")

            elif p == "gemini" and self.gemini_client:
                try:
                    m = model if (model and model.startswith("gemini-")) else getattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")
                    stream = self.gemini_client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        **kwargs
                    )
                    for chunk in stream:
                        content = chunk.choices[0].delta.content if (chunk.choices and chunk.choices[0].delta) else ""
                        if content:
                            yield content
                    return
                except Exception as e:
                    logger.warning(f"Gemini stream error: {e}...")

        yield " [Streaming failed across all available providers. Please check your network connection or API keys.]"
