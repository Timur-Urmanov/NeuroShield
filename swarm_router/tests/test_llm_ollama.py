"""OllamaProvider against a fake local /api/chat server (offline)."""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from swarm.llm.config import ModelSpec
from swarm.llm.providers import OllamaProvider, ProviderError

SEEN = []


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append(body)
        n = sum(len(m["content"]) for m in body["messages"]) // 4
        trunc = body["options"]["num_ctx"] < 1000
        out = dict(message={"role": "assistant", "content": "Final answer: 4"}, done=True,
                   done_reason="stop", prompt_eval_count=min(n, 200) if trunc else n, eval_count=5)
        b = json.dumps(out).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    s = HTTPServer(("127.0.0.1", 0), H); t = threading.Thread(target=s.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def spec(url, **params):
    return ModelSpec(name="q", provider="ollama", model="qwen2.5:0.5b", price_in_per_1m=0.5, price_out_per_1m=0.5,
                     base_url=url, params=params)


def test_ollama_roundtrip_and_seed(server):
    SEEN.clear()
    p = OllamaProvider(spec(server, num_ctx=16384, temperature=0.7))
    c = p.complete("sys", [{"role": "user", "content": "2+2?"}], 64, sample_id=3)
    assert c.text == "Final answer: 4" and c.output_tokens == 5 and c.stop_reason == "end_turn"
    o = SEEN[-1]["options"]
    assert o["num_ctx"] == 16384 and o["num_predict"] == 64 and SEEN[-1]["messages"][0]["role"] == "system"
    p.complete("sys", [{"role": "user", "content": "2+2?"}], 64, sample_id=3)
    p.complete("sys", [{"role": "user", "content": "2+2?"}], 64, sample_id=4)
    assert SEEN[-3]["options"]["seed"] == SEEN[-2]["options"]["seed"] != SEEN[-1]["options"]["seed"]


def test_ollama_requires_num_ctx_and_detects_truncation(server):
    with pytest.raises(ProviderError):
        OllamaProvider(spec(server)).complete(None, [{"role": "user", "content": "x"}], 8)
    with pytest.raises(ProviderError, match="truncated"):
        OllamaProvider(spec(server, num_ctx=512)).complete(None, [{"role": "user", "content": "y" * 20000}], 8)
