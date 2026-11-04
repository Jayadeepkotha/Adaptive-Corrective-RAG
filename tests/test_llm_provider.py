import pytest
from unittest.mock import MagicMock, patch
from src.llm import UnifiedLLMClient
from src.classifier import AdaptiveClassifier, ComplexityLevel
from src.generator import AnswerGenerator

def test_unified_llm_client_missing_keys():
    with patch("src.llm.settings.GROQ_API_KEY", ""), \
         patch("src.llm.settings.GEMINI_API_KEY", ""), \
         patch("src.llm.settings.GOOGLE_API_KEY", ""), \
         patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="No LLM API key configured"):
            UnifiedLLMClient(provider="groq", api_key="")

def test_unified_llm_client_groq_init():
    with patch("src.llm.settings.GROQ_API_KEY", "gsk_test123"), \
         patch("src.llm.settings.GEMINI_API_KEY", ""), \
         patch("src.llm.Groq") as mock_groq:
        client = UnifiedLLMClient()
        assert client.groq_client is not None
        assert client.gemini_client is None
        mock_groq.assert_called_once_with(api_key="gsk_test123", timeout=20.0)

def test_unified_llm_client_gemini_init():
    with patch("src.llm.settings.GROQ_API_KEY", ""), \
         patch("src.llm.settings.GEMINI_API_KEY", "AIzaSyTestKey"), \
         patch("src.llm.settings.LLM_PROVIDER", "gemini"), \
         patch.dict("os.environ", {}, clear=True), \
         patch("src.llm.OpenAI") as mock_openai:
        client = UnifiedLLMClient()
        assert client.groq_client is None
        assert client.gemini_client is not None
        mock_openai.assert_any_call(
            api_key="AIzaSyTestKey",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=8.0
        )

def test_unified_llm_client_dual_keys_auto_failover():
    with patch("src.llm.settings.GROQ_API_KEY", "gsk_test123"), \
         patch("src.llm.settings.GEMINI_API_KEY", "AIzaSyTestKey"), \
         patch("src.llm.settings.LLM_PROVIDER", "groq"), \
         patch.dict("os.environ", {}, clear=True), \
         patch("src.llm.Groq") as mock_groq_cls, \
         patch("src.llm.OpenAI") as mock_openai_cls:
        
        mock_groq_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_openai_cls.return_value = mock_gemini_instance
        
        # Groq fails with 429 rate limit
        mock_groq_instance.chat.completions.create.side_effect = RuntimeError("Rate limit 429: tokens per day exceeded")
        
        # Gemini succeeds
        gemini_resp = MagicMock()
        gemini_choice = MagicMock()
        gemini_choice.message.content = "Answer from Gemini"
        gemini_resp.choices = [gemini_choice]
        gemini_resp.usage.prompt_tokens = 25
        gemini_resp.usage.completion_tokens = 5
        mock_gemini_instance.chat.completions.create.return_value = gemini_resp
        
        client = UnifiedLLMClient(provider="groq")
        client.ollama_client = None
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert resp.choices[0].message.content == "Answer from Gemini"
        # Verified Groq was tried first, then Gemini was called
        assert mock_groq_instance.chat.completions.create.call_count >= 1
        assert mock_gemini_instance.chat.completions.create.call_count == 1

def test_unified_llm_client_gemini_provider_preference():
    with patch("src.llm.settings.GROQ_API_KEY", "gsk_test123"), \
         patch("src.llm.settings.GEMINI_API_KEY", "AIzaSyTestKey"), \
         patch("src.llm.Groq") as mock_groq_cls, \
         patch("src.llm.OpenAI") as mock_openai_cls:
        
        mock_groq_instance = MagicMock()
        mock_gemini_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_openai_cls.return_value = mock_gemini_instance

        gemini_resp = MagicMock()
        gemini_choice = MagicMock()
        gemini_choice.message.content = "Gemini Direct"
        gemini_resp.choices = [gemini_choice]
        mock_gemini_instance.chat.completions.create.return_value = gemini_resp

        client = UnifiedLLMClient(provider="gemini")
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert resp.choices[0].message.content == "Gemini Direct"
        # Groq should not even be called when Gemini is preferred and succeeds
        mock_groq_instance.chat.completions.create.assert_not_called()

def test_classifier_with_gemini_key():
    with patch("src.llm.settings.GROQ_API_KEY", ""), \
         patch("src.llm.settings.GEMINI_API_KEY", "AIzaSyTestKey"), \
         patch("src.llm.OpenAI") as mock_openai_cls:
        
        mock_gemini = MagicMock()
        mock_openai_cls.return_value = mock_gemini
        gemini_resp = MagicMock()
        gemini_choice = MagicMock()
        gemini_choice.message.content = '{"complexity": "SINGLE_HOP", "reasoning": "Simple question", "confidence": 0.99}'
        gemini_resp.choices = [gemini_choice]
        gemini_resp.usage.prompt_tokens = 15
        gemini_resp.usage.completion_tokens = 10
        mock_gemini.chat.completions.create.return_value = gemini_resp

        classifier = AdaptiveClassifier(provider="gemini")
        classifier.local_clf = None  # Force LLM execution for mock testing
        out = classifier.classify("What is the capital of France?")
        assert out.complexity == ComplexityLevel.SINGLE_HOP
        assert out.confidence == 0.99

def test_generator_with_gemini_key():
    with patch("src.llm.settings.GROQ_API_KEY", ""), \
         patch("src.llm.settings.GEMINI_API_KEY", "AIzaSyTestKey"), \
         patch("src.llm.OpenAI") as mock_openai_cls:
        
        mock_gemini = MagicMock()
        mock_openai_cls.return_value = mock_gemini
        gemini_resp = MagicMock()
        gemini_choice = MagicMock()
        gemini_choice.message.content = "Paris"
        gemini_resp.choices = [gemini_choice]
        gemini_resp.usage.prompt_tokens = 12
        gemini_resp.usage.completion_tokens = 2
        mock_gemini.chat.completions.create.return_value = gemini_resp

        generator = AnswerGenerator(provider="gemini")
        ans = generator.generate_direct("What is the capital of France?", eval_mode=True)
        assert ans == "Paris"
