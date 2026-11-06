
from backend import generator


class FakeOllamaClient:
    def chat(self, **kwargs):
        return {
            "message": {
                "content": "A database stores and manages organised data."
            }
        }


def test_generate_answer_returns_response(monkeypatch):
    monkeypatch.setattr(
        generator,
        "_client",
        FakeOllamaClient()
    )

    result = generator.generate_answer(
        "What is a database?",
        "A database stores and manages data."
    )

    assert isinstance(result, str)
    assert "database" in result.lower()


def test_empty_question_returns_message():
    result = generator.generate_answer("", "Some context.")

    assert result == "Please enter a question."


def test_empty_context_returns_message():
    result = generator.generate_answer(
        "What is a database?",
        ""
    )

    assert result == (
        "I could not find relevant information in the knowledge base."
    )


def test_whitespace_only_context_returns_message():
    result = generator.generate_answer(
        "What is a database?",
        "   "
    )

    assert result == (
        "I could not find relevant information in the knowledge base."
    )


class FakeStatusClient:
    def __init__(self, installed, loaded=()):
        self._installed = installed
        self._loaded = loaded

    def list(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            models=[SimpleNamespace(model=name) for name in self._installed]
        )

    def ps(self):
        from types import SimpleNamespace
        return SimpleNamespace(models=list(self._loaded))


def test_ollama_status_when_model_installed(monkeypatch):
    monkeypatch.setattr(generator, "LLM_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setattr(
        generator,
        "_status_client",
        FakeStatusClient(["qwen2.5:3b-instruct", "llama3.2:1b"])
    )

    status = generator.get_ollama_status()

    assert status["reachable"] is True
    assert status["model_installed"] is True
    assert status["processor"] is None


def test_ollama_status_when_model_missing(monkeypatch):
    monkeypatch.setattr(generator, "LLM_MODEL", "qwen2.5:3b-instruct")
    monkeypatch.setattr(
        generator, "_status_client", FakeStatusClient(["llama3.2:1b"])
    )

    status = generator.get_ollama_status()

    assert status["reachable"] is True
    assert status["model_installed"] is False


def test_ollama_status_accepts_implicit_latest_tag(monkeypatch):
    monkeypatch.setattr(generator, "LLM_MODEL", "llama3.2")
    monkeypatch.setattr(
        generator, "_status_client", FakeStatusClient(["llama3.2:latest"])
    )

    assert generator.get_ollama_status()["model_installed"] is True


def test_ollama_status_when_unreachable(monkeypatch):
    class DownClient:
        def list(self):
            raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(generator, "_status_client", DownClient())

    status = generator.get_ollama_status()

    assert status["reachable"] is False
    assert status["model_installed"] is False
    assert "Failed to connect" in status["error"]


def test_ollama_status_reports_processor(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(generator, "LLM_MODEL", "qwen2.5:3b-instruct")

    def status_for(size_vram):
        monkeypatch.setattr(
            generator,
            "_status_client",
            FakeStatusClient(
                ["qwen2.5:3b-instruct"],
                loaded=[SimpleNamespace(
                    model="qwen2.5:3b-instruct",
                    size=1000,
                    size_vram=size_vram
                )]
            )
        )
        return generator.get_ollama_status()["processor"]

    assert status_for(1000) == "GPU"
    assert status_for(0) == "CPU"
    assert status_for(600) == "60% GPU / 40% CPU"
