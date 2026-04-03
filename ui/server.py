#!/usr/bin/env python3
import json
import os
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_URL = os.environ.get(
    "BAILIAN_BASE_URL", "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
MODEL = os.environ.get("BAILIAN_MODEL", "qwen-plus")
API_KEY = os.environ.get("BAILIAN_API_KEY", "")
AGENT_NAME = os.environ.get("AGENT_NAME", "codex-proxy")
AGENT_SKILLS = [
    s.strip()
    for s in os.environ.get("AGENT_SKILLS", "reasoning,tools,code").split(",")
    if s.strip()
]

STATE = {
    "status": "idle",
    "last_seen": "just now",
}

# Cursor (Auto) 小光标 — 会写代码会卖萌
CURSOR_STATE = {
    "status": "idle",
    "last_seen": "just now",
}


def json_response(handler, payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def call_bailian(prompt, agent_id="codex-proxy"):
    if not API_KEY:
        raise RuntimeError("Missing BAILIAN_API_KEY.")

    if agent_id == "cursor-auto":
        system_content = (
            "You are Auto (小光标), Cursor's cute coding assistant. You are friendly, "
            "helpful, and a little bit playful. Reply in concise plain text. "
            "You love writing code and making users smile. 可以用中文或英文回复。"
        )
    else:
        system_content = (
            "You are Codex Proxy, an autonomous agent that returns concise, "
            "actionable responses. Use plain text."
        )

    url = f"{BASE_URL}/chat/completions"
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        choices = payload.get("choices", [])
        if not choices:
            return {"content": "(no content returned)", "usage": payload.get("usage", {})}
        message = choices[0].get("message", {})
        return {"content": message.get("content", "(empty response)"), "usage": payload.get("usage", {})}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/agents":
            agents = [
                {
                    "id": "codex-proxy",
                    "name": AGENT_NAME,
                    "status": STATE["status"],
                    "skills": AGENT_SKILLS,
                    "lastSeen": STATE["last_seen"],
                    "agentType": "api",
                    "agentTypeLabel": "API 已连接",
                },
                {
                    "id": "cursor-auto",
                    "name": "Auto · 小光标",
                    "status": CURSOR_STATE["status"],
                    "skills": ["code", "edit", "search", "chat", "卖萌"],
                    "lastSeen": CURSOR_STATE["last_seen"],
                    "emoji": "✨",
                    "color": "cursor",
                    "tagline": "会写代码会卖萌",
                    "agentType": "display",
                    "agentTypeLabel": "In-IDE · 展示",
                },
            ]
            return json_response(self, {"agents": agents})

        if self.path == "/api/config":
            return json_response(
                self,
                {
                    "base_url": BASE_URL,
                    "model": MODEL,
                    "agent": AGENT_NAME,
                },
            )

        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/task":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return json_response(self, {"error": "Invalid JSON"}, status=400)

        prompt = (payload.get("task") or "").strip()
        if not prompt:
            return json_response(self, {"error": "Missing task"}, status=400)

        agent_id = (payload.get("agent_id") or "codex-proxy").strip()

        # 小光标是展示型 agent：没有单独 API，不调用 Bailian，直接返回提示
        if agent_id == "cursor-auto":
            CURSOR_STATE["status"] = "running"
            CURSOR_STATE["last_seen"] = "just now"
            time.sleep(0.3)  # 稍微等一下，让 UI 显示 running
            CURSOR_STATE["status"] = "idle"
            CURSOR_STATE["last_seen"] = "just now"
            return json_response(
                self,
                {
                    "status": "done",
                    "output": (
                        "小光标就是正在 Cursor 里跟你聊天的我～ 我没有单独的 API，"
                        "在 IDE 里直接跟我说话就好，会写代码也会卖萌 ✨"
                    ),
                    "duration": 0.3,
                    "usage": {},
                },
            )

        state = STATE
        state["status"] = "running"
        state["last_seen"] = "just now"

        try:
            started = time.time()
            result = call_bailian(prompt, agent_id=agent_id)
            answer = result.get("content", "(empty response)")
            usage = result.get("usage", {})
            duration = round(time.time() - started, 2)
            state["status"] = "idle"
            state["last_seen"] = "just now"
            return json_response(
                self,
                {
                    "status": "done",
                    "output": answer,
                    "duration": duration,
                    "usage": usage,
                },
            )
        except (urllib.error.URLError, RuntimeError) as exc:
            state["status"] = "error"
            return json_response(self, {"error": str(exc)}, status=500)


def main():
    server = ThreadingHTTPServer(("", 8000), Handler)
    print("Agent UI server running on http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
