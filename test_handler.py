"""Worker tests against a simulated SGLang server and Cinevora (no GPU needed).

    python runpod_workers/minimax_h3/test_handler.py
"""
import json
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


SGL_PORT, WEB_PORT = free_port(), free_port()
os.environ["SGLANG_PORT"] = str(SGL_PORT)
os.environ["H3_MEDIA_DIR"] = str(Path(os.environ.get("TMPDIR", "/tmp")) / "h3-test-media")

import handler as h  # noqa: E402

STATE = {"bodies": [], "polls": 0, "uploads": {}, "fail_next": False}
VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 4096


class SGLang(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": "MiniMaxAI/MiniMax-H3"}]})
        if self.path.endswith("/content"):
            self.send_response(200); self.send_header("Content-Length", str(len(VIDEO))); self.end_headers()
            return self.wfile.write(VIDEO)
        if self.path.startswith("/v1/videos/"):
            STATE["polls"] += 1
            if STATE["fail_next"]:
                return self._json(200, {"id": "vid_1", "status": "failed", "error": {"message": "CUDA out of memory"}})
            return self._json(200, {"id": "vid_1", "status": "completed" if STATE["polls"] >= 2 else "in_progress"})
        self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["bodies"].append(body)
        for cond in body.get("conditions", []):  # SGLang reads local files: they must exist
            assert Path(cond["uri"][len("file://"):]).exists(), cond
        self._json(200, {"id": "vid_1", "status": "queued"})


class Web(BaseHTTPRequestHandler):
    """Cinevora: serves input media and accepts the one-time PUT upload."""
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/missing"):
            self.send_response(404); self.end_headers(); return
        data = b"media:" + self.path.encode() * 50
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_PUT(self):
        n = int(self.headers.get("Content-Length") or 0)
        STATE["uploads"][self.path] = self.rfile.read(n)
        self.send_response(204); self.end_headers()


for port, cls in ((SGL_PORT, SGLang), (WEB_PORT, Web)):
    srv = ThreadingHTTPServer(("127.0.0.1", port), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

h.start_server = lambda: None  # the fake server is already running
h._served_model = "MiniMaxAI/MiniMax-H3"
WEB = f"http://127.0.0.1:{WEB_PORT}"
FAILURES = []


def check(label, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def run(inp, variant):
    h.VARIANT = variant
    STATE["polls"] = 0
    return h.handler({"id": "job", "input": inp})


# text-to-video on the FL2VA worker
out = run({"task": "t2va", "prompt": "A lighthouse at dawn", "seconds": 5, "short_edge": 768,
           "aspect_ratio": "16:9", "seed": 7, "output_upload_url": f"{WEB}/__upscale_upload__/tok1"}, "fl2va")
body = STATE["bodies"][-1]
check("t2va succeeds", out.get("ok") is True, out)
check("t2va request matches the cookbook shape",
      body["task"] == "t2va" and body["conditions"] == [] and body["seconds"] == 5
      and body["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5}
      and body["seed"] == 7 and body["num_inference_steps"] == 50 and body["model"] == "MiniMaxAI/MiniMax-H3", body)
check("video uploaded to the one-time URL", STATE["uploads"].get("/__upscale_upload__/tok1") == VIDEO)
check("timings reported", set(out.get("timings", {})) == {"inputs", "generate", "upload", "total"}, out)

# first-frame image-to-video
out = run({"task": "fl2va", "prompt": "Continue the shot", "seconds": 8, "short_edge": 480, "aspect_ratio": "auto",
           "keyframes": [{"url": f"{WEB}/frame.png", "frame_index": 0}],
           "output_upload_url": f"{WEB}/__upscale_upload__/tok2"}, "fl2va")
body = STATE["bodies"][-1]
check("fl2va keyframe sent as a local file at frame 0",
      out.get("ok") and len(body["conditions"]) == 1 and body["conditions"][0]["role"] == "keyframe"
      and body["conditions"][0]["frame_index"] == 0 and body["conditions"][0]["uri"].startswith("file://")
      and body["target"]["aspect_ratio"] == "auto", (out, body))

# omni-reference with characters' images and voices
out = run({"task": "ref2va", "prompt": "<Subject 1> greets <Subject 2>.", "seconds": 10, "short_edge": 768,
           "aspect_ratio": "16:9", "images": [f"{WEB}/leroy.png", f"{WEB}/goy.png"],
           "audio": [f"{WEB}/leroy.wav"], "output_upload_url": f"{WEB}/__upscale_upload__/tok3"}, "ref2va")
body = STATE["bodies"][-1]
kinds = [(c["type"], c["role"]) for c in body["conditions"]]
check("ref2va sends images then audio as references",
      out.get("ok") and kinds == [("image", "reference"), ("image", "reference"), ("audio", "reference")], (out, kinds))
check("random seed chosen when none is given", isinstance(body["seed"], int) and body["seed"] > 0)

# validation and errors (no generation attempted)
n = len(STATE["bodies"])
cases = [
    ({"task": "ref2va", "prompt": "x", "images": [f"{WEB}/a.png"], "output_upload_url": f"{WEB}/u"}, "fl2va", "cannot run task"),
    ({"task": "t2va", "prompt": "x", "seconds": 20, "output_upload_url": f"{WEB}/u"}, "fl2va", "between 4 and 15"),
    ({"task": "t2va", "prompt": "x", "short_edge": 1080, "output_upload_url": f"{WEB}/u"}, "fl2va", "768"),
    ({"task": "t2va", "prompt": "x"}, "fl2va", "output_upload_url"),
    ({"task": "ref2va", "prompt": "x", "images": [f"{WEB}/{i}.png" for i in range(10)], "output_upload_url": f"{WEB}/u"}, "ref2va", "max 9"),
    ({"task": "ref2va", "prompt": "x", "images": [f"{WEB}/missing.png"], "output_upload_url": f"{WEB}/u"}, "ref2va", "HTTP 404"),
    ({"task": "fl2va", "prompt": "x", "output_upload_url": f"{WEB}/u"}, "fl2va", "one or two keyframes"),
]
for inp, variant, expect in cases:
    out = run(inp, variant)
    check(f"rejects: {expect}", out.get("ok") is False and expect in out.get("error", ""), out)
check("rejected jobs never reach SGLang", len(STATE["bodies"]) == n)

STATE["fail_next"] = True
out = run({"task": "t2va", "prompt": "x", "output_upload_url": f"{WEB}/u"}, "fl2va")
STATE["fail_next"] = False
check("generation failure is reported with SGLang's message", out.get("ok") is False and "CUDA out of memory" in out["error"], out)

check("temporary media is cleaned up", not any(Path(os.environ["H3_MEDIA_DIR"]).glob("job-*")))
check("default placement for one B200", h.default_sglang_args([("NVIDIA B200", 183000)]) == "--performance-mode speed")
check("default placement for 4x H100 matches the cookbook",
      h.default_sglang_args([("NVIDIA H100 80GB HBM3", 81559)] * 4) == "--num-gpus 4 --tp-size 2 --ulysses-degree 2 --performance-mode speed")

print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}")
sys.exit(1 if FAILURES else 0)
