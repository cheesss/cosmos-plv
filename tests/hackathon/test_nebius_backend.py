import json
import types

import pytest

openai = pytest.importorskip("openai")

from cosmos_plv import nebius_backend as nb  # noqa: E402

FAKE_KEY = "v1.FAKE-KEY-FOR-TESTS-ONLY"


def test_key_from_env_and_file(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_API_KEY", FAKE_KEY)
    assert nb.load_api_key() == FAKE_KEY
    monkeypatch.delenv("NEBIUS_API_KEY")
    f = tmp_path / "nebius.env"
    f.write_text(f'NEBIUS_API_KEY="{FAKE_KEY}"\n')
    monkeypatch.setattr(nb, "SECRET_FILES", [str(f)])
    assert nb.load_api_key() == FAKE_KEY


def test_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(nb, "SECRET_FILES", [])
    with pytest.raises(RuntimeError):
        nb.NebiusVLMClient()


class _FakeCompletions:
    def __init__(self, fail_first):
        self.calls = 0
        self.fail_first = fail_first

    def create(self, **kw):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("404 not found")
        msg = types.SimpleNamespace(content='{"ok": true}')
        return types.SimpleNamespace(
            id="chatcmpl-test", model=kw["model"], choices=[types.SimpleNamespace(message=msg)],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=3))


def _client(tmp_path, fail_first=False):
    c = nb.NebiusVLMClient(api_key=FAKE_KEY, evidence_path=str(tmp_path / "ev.jsonl"))
    comp = _FakeCompletions(fail_first)
    c._client = lambda url: types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    return c, comp


def test_evidence_log_has_no_key_and_records_usage(tmp_path):
    c, _ = _client(tmp_path)
    r = c.generate("hello", tag="t")
    assert r.raw_text == '{"ok": true}' and r.request_id == "chatcmpl-test" and r.error is None
    text = (tmp_path / "ev.jsonl").read_text()
    assert FAKE_KEY not in text
    rec = json.loads(text.splitlines()[0])
    assert rec["prompt_tokens"] == 10 and rec["tag"] == "t" and "raw_text" not in rec


def test_regional_fallback(tmp_path):
    c, comp = _client(tmp_path, fail_first=True)
    r = c.generate("hello")
    assert r.error is None and comp.calls == 2 and r.params["base_url"] == nb.BASE_URLS[1]
