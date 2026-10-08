"""RunPod serverless worker for self-hosted MiniMax H3 (open weights, H3-Base).

One worker serves one checkpoint variant, chosen with H3_VARIANT:
  fl2va   text-to-video and first/last-frame image-to-video   (tasks t2va, fl2va)
  ref2va  omni-reference: up to 9 images, 3 videos, 3 audio  (task ref2va)

Each worker starts a local SGLang server on boot, then for every job:
  1. downloads the input media URLs to local files,
  2. calls SGLang POST /v1/videos with H3's request format,
  3. waits for the video, downloads it,
  4. uploads it to the one-time PUT URL Cinevora sent (output_upload_url),
  5. returns {"ok": true, ...} or {"ok": false, "error": "..."}.

The video never passes through RunPod's job output, so there is no payload
size limit to worry about. Licensed to Naticis under MiniMax's authorization;
the MiniMax H3 Community License terms (attribution, acceptable use, no
distillation) still apply to everything this worker generates.
"""
import json
import mimetypes
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------- configuration
VARIANT = os.getenv("H3_VARIANT", "fl2va").strip().lower()
MODEL_PATH = os.getenv("H3_MODEL_PATH", "/runpod-volume/MiniMax-H3")
HF_REPO = os.getenv("H3_HF_REPO", "MiniMaxAI/MiniMax-H3")
AUTO_DOWNLOAD = os.getenv("H3_AUTO_DOWNLOAD", "0") == "1"
PORT = int(os.getenv("SGLANG_PORT", "30010"))
BASE = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT = int(os.getenv("SGLANG_STARTUP_TIMEOUT", "2400"))     # first load reads ~100 GB
JOB_TIMEOUT = int(os.getenv("H3_JOB_TIMEOUT", "1800"))
MEDIA_DIR = Path(os.getenv("H3_MEDIA_DIR", "/tmp/h3-media"))
MAX_INPUT_BYTES = int(os.getenv("H3_MAX_INPUT_BYTES", str(200 * 1024 * 1024)))
DEFAULT_STEPS = int(os.getenv("H3_DEFAULT_STEPS", "50"))
DEFAULT_QUALITY = os.getenv("H3_DEFAULT_QUALITY", "exact")

TASKS_BY_VARIANT = {"fl2va": {"t2va", "fl2va"}, "ref2va": {"ref2va"}}
VARIANT_DIRS = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}
ASPECT_RATIOS = {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "auto"}
SHORT_EDGES = {480, 768}
REF_LIMITS = {"images": 9, "videos": 3, "audio": 3, "total": 12}

_server = None
_served_model = None


def log(msg):
    print(f"[h3-worker] {msg}", flush=True)


# ---------------------------------------------------------------- SGLang server
def gpu_inventory():
    """[(name, memory_mib), ...] from nvidia-smi; empty when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, _, mem = line.rpartition(",")
        try:
            gpus.append((name.strip(), int(mem.strip())))
        except ValueError:
            pass
    return gpus


def default_sglang_args(gpus):
    """Placement flags from the SGLang H3 cookbook. Override with SGLANG_ARGS."""
    count = len(gpus)
    min_mem = min((m for _, m in gpus), default=0)
    name = (gpus[0][0] if gpus else "").lower()
    if count >= 4:
        if "h100" in name:  # cookbook: H100 x4 resident
            return f"--num-gpus {count} --tp-size 2 --ulysses-degree {count // 2} --performance-mode speed"
        return f"--num-gpus {count} --ulysses-degree {count} --performance-mode speed"
    if count >= 2:
        return f"--num-gpus {count} --ulysses-degree {count} --performance-mode speed"
    if min_mem >= 150_000:  # B200-class: full model resident on one card
        return "--performance-mode speed"
    if min_mem >= 90_000:   # cookbook: RTX PRO 6000 96 GB (derived recipe)
        return ("--performance-mode memory --layerwise-offload-components text_encoder,vae "
                "--layerwise-resident-layers video_vae=36")
    # Smaller cards: full layerwise offload (slow; the cookbook's consumer recipe).
    return "--performance-mode memory --layerwise-offload-components dit,text_encoder,vae"


def download_weights(root, patterns):
    """Download matching files of the H3 repo into root (resumes partial files).

    Uses the Python API rather than the `hf` command: the command's --include
    option changed between huggingface_hub versions (one pattern per flag in
    1.x), which made it treat "FL2VA/*" as a file name.
    """
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, local_dir=str(root), allow_patterns=list(patterns),
                      max_workers=int(os.getenv("H3_DOWNLOAD_WORKERS", "16")))


def _weights_marker():
    return Path(MODEL_PATH) / f".download-complete-{VARIANT_DIRS[VARIANT]}"


def ensure_weights():
    """Make sure this variant's checkpoint is on the volume.

    With H3_AUTO_DOWNLOAD=1 the worker downloads it itself (one time, ~144 GB).
    A marker file is written only after a download finishes, so an interrupted
    download is resumed by the next worker instead of being mistaken for a
    complete one. Without auto-download, a manually downloaded folder is used.
    """
    root = Path(MODEL_PATH)
    have = (root / "model_index.json").exists() and (root / VARIANT_DIRS[VARIANT]).exists()
    if not AUTO_DOWNLOAD:
        if have:
            return
        raise JobError(
            f"H3 weights not found at {root} (need model_index.json and {VARIANT_DIRS[VARIANT]}/). "
            "Download them to the network volume first, or set H3_AUTO_DOWNLOAD=1."
        )
    if have and _weights_marker().exists():
        return
    log(f"downloading {HF_REPO} {VARIANT_DIRS[VARIANT]} to {root} "
        f"({'resuming' if have else 'one time, ~144 GB'})...")
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        download_weights(root, ["model_index.json", f"{VARIANT_DIRS[VARIANT]}/*"])
    except Exception as exc:
        raise JobError(f"Weight download failed ({type(exc).__name__}: {exc}); the next start resumes it.")
    _weights_marker().write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"download complete in {time.time() - started:.0f}s")


def start_server():
    """Start SGLang once per worker and wait until it answers."""
    global _server, _served_model
    if _server is not None and _server.poll() is None:
        return
    if VARIANT not in TASKS_BY_VARIANT:
        raise RuntimeError(f"H3_VARIANT must be fl2va or ref2va, not {VARIANT!r}")
    ensure_weights()
    gpus = gpu_inventory()
    extra = os.getenv("SGLANG_ARGS", "").strip() or default_sglang_args(gpus)
    cmd = (["sglang", "serve", "--model-path", MODEL_PATH, "--model-variant", VARIANT,
            "--host", "127.0.0.1", "--port", str(PORT)] + shlex.split(extra))
    log(f"GPUs: {gpus or 'unknown'}")
    log("starting: " + " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _server = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup with code {_server.returncode}")
        try:
            r = requests.get(f"{BASE}/v1/models", timeout=5)
            if r.ok:
                models = (r.json() or {}).get("data") or []
                _served_model = (models[0].get("id") if models else None) or MODEL_PATH
                log(f"SGLang ready in {int(time.time() - (deadline - STARTUP_TIMEOUT))}s; model id {_served_model}")
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"SGLang did not become ready within {STARTUP_TIMEOUT}s")


# ---------------------------------------------------------------- job input
class JobError(Exception):
    pass


def _int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate(job_input):
    inp = dict(job_input or {})
    task = str(inp.get("task") or "").lower()
    if task not in TASKS_BY_VARIANT[VARIANT]:
        raise JobError(f"This worker runs the {VARIANT} checkpoint and cannot run task {task!r}.")
    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise JobError("A prompt is required.")
    seconds = _int(inp.get("seconds"), 5)
    if not 4 <= seconds <= 15:
        raise JobError("seconds must be between 4 and 15.")
    short_edge = _int(inp.get("short_edge"), 768)
    if short_edge not in SHORT_EDGES:
        raise JobError("short_edge must be 480 or 768 (self-hosted H3 stops at 768).")
    aspect = str(inp.get("aspect_ratio") or "16:9")
    if aspect not in ASPECT_RATIOS:
        raise JobError(f"Unsupported aspect_ratio {aspect!r}.")
    upload = str(inp.get("output_upload_url") or "")
    if not upload.startswith(("https://", "http://")):
        raise JobError("output_upload_url is required.")
    keyframes = [k for k in (inp.get("keyframes") or []) if isinstance(k, dict) and k.get("url")]
    images = [u for u in (inp.get("images") or []) if u]
    videos = [u for u in (inp.get("videos") or []) if u]
    audio = [u for u in (inp.get("audio") or []) if u]
    if task == "t2va" and (keyframes or images or videos or audio):
        raise JobError("t2va takes no media; use fl2va or ref2va.")
    if task == "fl2va":
        if not 1 <= len(keyframes) <= 2 or images or videos or audio:
            raise JobError("fl2va takes one or two keyframes (first and/or last frame) and nothing else.")
        if any(_int(k.get("frame_index"), 0) not in (0, -1) for k in keyframes):
            raise JobError("keyframe frame_index must be 0 (first) or -1 (last).")
    if task == "ref2va":
        if keyframes:
            raise JobError("ref2va takes references, not keyframes.")
        if not (images or videos or audio):
            raise JobError("ref2va needs at least one reference.")
        for kind, items in (("images", images), ("videos", videos), ("audio", audio)):
            if len(items) > REF_LIMITS[kind]:
                raise JobError(f"Too many {kind} references: {len(items)} (max {REF_LIMITS[kind]}).")
        if len(images) + len(videos) + len(audio) > REF_LIMITS["total"]:
            raise JobError(f"Too many references in total (max {REF_LIMITS['total']}).")
    seed = _int(inp.get("seed"), -1)
    if seed < 0:
        seed = random.randint(1, 2**31 - 1)
    return {
        "task": task, "prompt": prompt, "seconds": seconds, "short_edge": short_edge,
        "aspect_ratio": aspect, "seed": seed, "upload": upload, "keyframes": keyframes,
        "images": images, "videos": videos, "audio": audio,
        "steps": max(1, min(100, _int(inp.get("num_inference_steps"), DEFAULT_STEPS))),
        "quality": str(inp.get("quality") or DEFAULT_QUALITY),
    }


def fetch(url, folder, label):
    """Download one input URL to a local file (size-capped) and return its path."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise JobError(f"{label}: only http(s) URLs are accepted.")
    suffix = Path(parsed.path).suffix.lower()[:8]
    with requests.get(url, stream=True, timeout=(15, 120)) as r:
        if not r.ok:
            raise JobError(f"{label}: download failed with HTTP {r.status_code}.")
        if not suffix:
            suffix = mimetypes.guess_extension((r.headers.get("Content-Type") or "").split(";")[0].strip()) or ".bin"
        path = folder / f"{label}{suffix}"
        size = 0
        with path.open("wb") as fh:
            for chunk in r.iter_content(1024 * 1024):
                size += len(chunk)
                if size > MAX_INPUT_BYTES:
                    raise JobError(f"{label}: larger than {MAX_INPUT_BYTES // (1024 * 1024)} MB.")
                fh.write(chunk)
    if size == 0:
        raise JobError(f"{label}: empty file.")
    return path


def build_request(job, local):
    """The /v1/videos body from the SGLang MiniMax-H3 cookbook."""
    conditions = []
    if job["task"] == "fl2va":
        for kf, path in zip(job["keyframes"], local["keyframes"]):
            conditions.append({"type": "image", "uri": path.as_uri(), "role": "keyframe",
                               "frame_index": _int(kf.get("frame_index"), 0)})
    elif job["task"] == "ref2va":
        conditions += [{"type": "image", "uri": p.as_uri(), "role": "reference"} for p in local["images"]]
        conditions += [{"type": "video", "uri": p.as_uri(), "role": "reference", "start_time_seconds": 0}
                       for p in local["videos"]]
        conditions += [{"type": "audio", "uri": p.as_uri(), "role": "reference"} for p in local["audio"]]
    aspect = job["aspect_ratio"]
    if aspect == "auto" and not conditions:
        aspect = "16:9"  # "auto" follows the input media; text-only needs a real ratio
    return {
        "model": _served_model or MODEL_PATH,
        "prompt": job["prompt"],
        "seconds": job["seconds"],
        "task": job["task"],
        "conditions": conditions,
        "target": {
            "short_edge": job["short_edge"],
            "aspect_ratio": aspect,
            "duration_seconds": job["seconds"],
        },
        "quality": job["quality"],
        "num_outputs_per_prompt": 1,
        "num_inference_steps": job["steps"],
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": job["seed"],
    }


# ---------------------------------------------------------------- SGLang job
def _status_of(obj):
    return str((obj or {}).get("status") or "").lower()


def run_generation(body):
    r = requests.post(f"{BASE}/v1/videos", json=body, timeout=120)
    if not r.ok:
        raise JobError(f"SGLang rejected the request (HTTP {r.status_code}): {r.text[:500]}")
    created = r.json() if r.content else {}
    video_id = created.get("id")
    if not video_id:
        raise JobError(f"SGLang returned no video id: {str(created)[:300]}")
    status = _status_of(created)
    deadline = time.time() + JOB_TIMEOUT
    while status not in ("completed", "succeeded", "success"):
        if status in ("failed", "error", "cancelled", "canceled"):
            err = created.get("error")
            raise JobError(f"Generation failed: {err.get('message') if isinstance(err, dict) else err or status}")
        if time.time() > deadline:
            raise JobError(f"Generation did not finish within {JOB_TIMEOUT}s.")
        if _server is not None and _server.poll() is not None:
            raise RuntimeError("SGLang stopped during generation.")
        time.sleep(3)
        created = retrieve(video_id) or created
        status = _status_of(created)
    return video_id


def retrieve(video_id):
    """GET /v1/videos/{id}; falls back to the list endpoint the SGLang docs use."""
    try:
        r = requests.get(f"{BASE}/v1/videos/{video_id}", timeout=30)
        if r.ok:
            return r.json()
        if r.status_code not in (404, 405):
            return None
        r = requests.get(f"{BASE}/v1/videos", timeout=30)
        if r.ok:
            for item in (r.json() or {}).get("data") or []:
                if item.get("id") == video_id:
                    return item
    except requests.RequestException:
        return None
    return None


def download_result(video_id, folder):
    path = folder / "output.mp4"
    with requests.get(f"{BASE}/v1/videos/{video_id}/content", stream=True, timeout=(15, 600)) as r:
        if not r.ok:
            raise JobError(f"Could not download the generated video (HTTP {r.status_code}).")
        with path.open("wb") as fh:
            for chunk in r.iter_content(1024 * 1024):
                fh.write(chunk)
    if path.stat().st_size < 1024:
        raise JobError("The generated video is empty or invalid.")
    return path


def upload(path, url):
    last = None
    for attempt in range(1, 4):
        try:
            with path.open("rb") as fh:
                r = requests.put(url, data=fh, headers={"Content-Type": "video/mp4"}, timeout=(15, 900))
            if r.status_code in (200, 201, 204):
                return
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 403, 404):  # one-time URL already used or refused: do not retry
                break
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(3 * attempt)
    raise JobError(f"Could not upload the video back to Cinevora ({last}).")


# ---------------------------------------------------------------- handler
def handler(job):
    started = time.time()
    job_input = job.get("input") if isinstance(job, dict) else None
    folder = None
    try:
        if isinstance(job_input, dict) and job_input.get("warmup"):
            # {"input": {"warmup": true}}: download the weights if needed and load
            # the model, without generating. Handy right after deployment.
            start_server()
            return {"ok": True, "warmup": True, "variant": VARIANT, "model": _served_model,
                    "seconds": round(time.time() - started, 1)}
        start_server()
        spec = validate(job_input)
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f"job-{uuid.uuid4().hex[:8]}-", dir=MEDIA_DIR))
        local = {
            "keyframes": [fetch(k["url"], folder, f"keyframe{i + 1}") for i, k in enumerate(spec["keyframes"])],
            "images": [fetch(u, folder, f"image{i + 1}") for i, u in enumerate(spec["images"])],
            "videos": [fetch(u, folder, f"video{i + 1}") for i, u in enumerate(spec["videos"])],
            "audio": [fetch(u, folder, f"audio{i + 1}") for i, u in enumerate(spec["audio"])],
        }
        t_inputs = time.time()
        body = build_request(spec, local)
        log(f"task={spec['task']} seconds={spec['seconds']} short_edge={spec['short_edge']} "
            f"refs={len(body['conditions'])} seed={spec['seed']}")
        video_id = run_generation(body)
        t_generated = time.time()
        result = download_result(video_id, folder)
        size = result.stat().st_size
        upload(result, spec["upload"])
        done = time.time()
        return {
            "ok": True, "task": spec["task"], "seed": spec["seed"], "bytes": size,
            "timings": {"inputs": round(t_inputs - started, 2), "generate": round(t_generated - t_inputs, 2),
                        "upload": round(done - t_generated, 2), "total": round(done - started, 2)},
        }
    except JobError as exc:
        log(f"job failed: {exc}")
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # server crash or unexpected error: replace this worker
        log(f"worker error: {exc!r}")
        return {"ok": False, "error": f"Worker error: {exc}", "refresh_worker": True}
    finally:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)


def _shutdown(*_):
    if _server is not None and _server.poll() is None:
        _server.send_signal(signal.SIGTERM)
    sys.exit(0)


if __name__ == "__main__":
    import runpod  # imported here so the module can be tested without the RunPod SDK

    signal.signal(signal.SIGTERM, _shutdown)
    if os.getenv("H3_START_ON_BOOT", "1") == "1":
        # Load the model before taking the first job. If that fails (weights not
        # downloaded yet, wrong variant), keep the worker up instead of crashing:
        # a crash makes RunPod restart it in a loop and bill GPU time each time.
        # Jobs then retry the start and report the reason.
        try:
            start_server()
        except Exception as exc:
            log(f"model not loaded at boot: {exc}")
    runpod.serverless.start({"handler": handler})
