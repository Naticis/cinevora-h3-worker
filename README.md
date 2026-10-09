# MiniMax H3 Naticis — RunPod serverless worker

This folder builds the RunPod worker behind the **MiniMax H3 Naticis** model in
Cinevora. The worker runs the open MiniMax H3 weights with SGLang and sends each finished
video straight back to your Cinevora server through a one-time upload link.

Your use of the weights is covered by MiniMax's written authorization to Naticis, which
depends on you keeping the commitments in your request email. See "License duties" below.

## How it fits together

```
Cinevora (your VPS)                      RunPod
  make_movie.py ── /run ───────────────▶ endpoint "h3-fl2va"  (text-to-video, first/last frame)
                └─ /run ───────────────▶ endpoint "h3-ref2va" (characters / reference images, video, audio)
  /__upscale_upload__/<token> ◀── PUT ─── worker uploads the finished .mp4
```

There are two endpoints because H3 ships as two checkpoints (FL2VA and Ref2VA) and one
GPU worker cannot hold both. Cinevora picks the endpoint per scene:

| Scene in Cinevora | H3 task | Endpoint |
|---|---|---|
| Prompt only | `t2va` | FL2VA |
| `@image1` as first frame (or first + last) | `fl2va` | FL2VA |
| Characters, reference images, video or audio | `ref2va` | Ref2VA |

You can deploy only FL2VA at first. Scenes that need Ref2VA are then blocked before any
credits are taken, with a message saying the endpoint is not configured.

## 1. Put the weights on a network volume

Each variant folder in the repo is a complete checkpoint of about **144 GB**:

| You download | Size | Volume to create |
|---|---|---|
| FL2VA only (text-to-video, first/last frame) | ~144 GB | **200 GB** |
| FL2VA + Ref2VA (adds characters / references) | ~290 GB | **400 GB** |

A volume can be enlarged later but never shrunk, so starting with 200 GB and growing it
when you add Ref2VA is fine.

**Option A — let the serverless worker download them (no pod).** Attach the volume to the
endpoint, add the environment variable `H3_AUTO_DOWNLOAD=1`, set Max workers to 1, and send
one warmup request from the endpoint's Requests tab:

```json
{"input": {"warmup": true}, "policy": {"executionTimeout": 7200000}}
```

(`policy.executionTimeout` gives this one request 2 hours instead of the endpoint's
30 minutes, because it waits for the whole download. The worker reports ready to RunPod
at once and downloads in the background, so a slow download is not cut off.)

The worker downloads FL2VA (~144 GB) to the volume, loads the model and answers
`{"ok": true, "warmup": true, ...}`. The download runs on the GPU worker, so it is billed
at GPU rates for those minutes; if it is interrupted, the next start resumes it. Later
starts find the finished download and skip straight to loading.

**Option B — download from a cheap CPU pod** (slower to set up, cheaper per minute):

1. RunPod → Storage → New Network Volume. Pick a data center that offers the GPU you
   want for serverless (B200 or H200) — serverless workers can only use volumes in
   their own data center.
2. Deploy a cheap CPU pod with that volume attached (it mounts at `/workspace`), open
   its web terminal and run:

   ```bash
   pip install -U huggingface_hub
   python -c "from huggingface_hub import snapshot_download as d; d('MiniMaxAI/MiniMax-H3', local_dir='/workspace/MiniMax-H3', allow_patterns=['model_index.json', 'FL2VA/*'], max_workers=16)"
   # later, for characters/references: same command with allow_patterns=['Ref2VA/*']
   du -sh /workspace/MiniMax-H3/*      # FL2VA should be ~144G

   (The `hf download` command changed in huggingface_hub 1.x: `--include` takes one pattern
   per flag. The Python call above works on every version.)
   ```

   If the download stops, run the same command again; it resumes. Terminate the pod
   when done (the volume keeps the files).

Serverless workers see the same volume at `/runpod-volume`, which matches the default
`H3_MODEL_PATH=/runpod-volume/MiniMax-H3`. The other top-level folders in the repo
(`transformer/`, `vae/`, …) are a separate diffusers-format copy and are not needed.

## 2. Build the image

**Easiest: let RunPod build it from GitHub** (no Docker needed on your side).

1. Create a GitHub repository (private is fine), e.g. `cinevora-h3-worker`, and upload
   the files from this folder to its root: `Dockerfile`, `handler.py`, `README.md`,
   `NOTICE`, `test_handler.py` (GitHub → Add file → Upload files).
2. RunPod → Settings → Connections → GitHub → Connect, and give it access to that repo.
3. On GitHub, create a release (Releases → Draft a new release → tag `v1`). RunPod
   builds a new image only when a release is published.

RunPod's GitHub builds have limits: 30 minutes for `docker build`, 80 GB image size.
The SGLang base image is large; if the build times out, build it yourself instead:

```bash
docker build --platform linux/amd64 -t <dockerhub-user>/cinevora-h3:1 .
docker push <dockerhub-user>/cinevora-h3:1
```

`lmsysorg/sglang:dev` moves. After a build that works, pin it by digest in the
`Dockerfile` so a later rebuild cannot break silently.

You can test the worker logic without a GPU: `python test_handler.py`.

## 3. Create the endpoints

RunPod → Serverless → New Endpoint → **Import Git Repository** → your repo, branch `main`,
Dockerfile path `Dockerfile` (or **Docker Image** if you pushed one). Start with the FL2VA
endpoint; create the Ref2VA one the same way later, changing only `H3_VARIANT`:

| Setting | Value |
|---|---|
| Source | your GitHub repo (Import Git Repository), or the Docker image you pushed |
| Network volume | the volume from step 1 |
| GPU | **B200 (180 GB)** — one GPU, simplest. H200 ×4 / H100 ×4 also work (see below). |
| Env `H3_VARIANT` | `fl2va` on one endpoint, `ref2va` on the other |
| Execution timeout | 1800 s |
| Idle timeout | 60–300 s (longer = fewer cold starts, more idle cost) |
| Active workers | 0 to start; 1 if you want no cold starts |
| Max workers | how many videos you want rendering at once |
| FlashBoot | on |
| Container disk | 40 GB |

The worker reads its GPUs at start-up and picks SGLang settings from the MiniMax cookbook:
one large GPU → `--performance-mode speed`; 4× H100 → `--num-gpus 4 --tp-size 2
--ulysses-degree 2`; 4× H200 → `--ulysses-degree 4`; a 96 GB card → memory mode with
offloading. To override, set `SGLANG_ARGS` (passed straight to `sglang serve`).

Cold start loads ~144 GB from the volume, which can take several minutes. The first job
after an idle period waits for that; later jobs on the same worker do not.

### Worker settings (optional)

| Env | Default | Meaning |
|---|---|---|
| `H3_VARIANT` | `fl2va` | `fl2va` or `ref2va` |
| `H3_MODEL_PATH` | `/runpod-volume/MiniMax-H3` | Where the weights are |
| `H3_AUTO_DOWNLOAD` | `0` | `1`: the worker downloads missing weights itself (resumable) |
| `SGLANG_ARGS` | auto | Extra `sglang serve` arguments |
| `SGLANG_STARTUP_TIMEOUT` | `2400` | Seconds to wait for the model to load |
| `H3_JOB_TIMEOUT` | `1800` | Seconds one generation may take |
| `H3_DEFAULT_STEPS` | `50` | Denoising steps |
| `H3_MAX_INPUT_BYTES` | 200 MB | Largest reference file the worker will download |

## 4. Connect Cinevora

Add to the server environment (same place as the upscaler settings) and restart:

```
RUNPOD_API_KEY=<your RunPod API key>
RUNPOD_H3_FL2VA_ENDPOINT_ID=<endpoint id>
RUNPOD_H3_REF2VA_ENDPOINT_ID=<endpoint id>          # optional at first
KINOVI_PUBLIC_ASSET_BASE=https://your-domain        # already set if the upscaler works
ANTHROPIC_API_KEY=<Claude API key>                  # required: content screening (see CONTENT_SAFETY.txt)
KINOVI_ABUSE_EMAIL=abuse@your-domain                # shown on the Terms and Report pages
# optional
RUNPOD_H3_EXECUTION_TIMEOUT=1800
KINOVI_H3_OPEN_RATES={"480p": 8, "768p": 12}       # estimate rates, per second
```

**MiniMax H3 Naticis** appears in the model list only once at least one endpoint ID
is set. The worker needs to reach your server over HTTPS to download references and
upload results — the same path the upscaler already uses.

No provider API key is needed for projects that run only on this model. Projects that mix
models, or use continue/extend/upscale-extend/360/image-to-video modes, still need one.

## 5. Set your price

The credit prices in Admin → Pricing (`minimax-h3-open|480p` = 8, `|768p` = 12 per
second, before the 0.1 multiplier) are **placeholders**. Calibrate them:

1. Render ~10 scenes at each resolution and length you sell.
2. In RunPod billing, read the cost per job (GPU $/s × execution time; add cold starts).
3. Price = cost per second × (1 + your margin), converted at
   1 provider credit ≈ $10 / 2150.
4. Set those numbers in Admin → Pricing, and the same values in `KINOVI_H3_OPEN_RATES`
   so the batch estimate matches.

As a rough guide, a B200 at ~$6–9/hour rendering a 5 s 768p clip in a few minutes costs
on the order of $0.30–1.00 per clip before idle time — check against your own runs.

## What the model can take

* 4–15 seconds, 24 fps, 480p or 768p (short edge), aspect 21:9, 16:9, 4:3, 1:1, 3:4, 9:16.
* Ref2VA: up to 9 images, 3 videos, 3 audio clips, 12 in total. Cinevora checks this
  before charging.
* Prompts: Cinevora rewrites character names and `@imageN` to H3's `<Subject N>`,
  `<Video N>`, `<Audio N>` tags. This mapping follows the model card; confirm it on your
  first real renders and adjust `apply_h3_prompt_tags` in `make_movie.py` if needed.
* First-frame scenes run on FL2VA, which takes no reference images, so characters in
  such a scene are described in text only.

## License duties (MiniMax H3 Community License + your authorization)

Your authorization depends on the commitments you confirmed on the request form. How Cinevora covers them:

| Commitment | In Cinevora (v53) |
|---|---|
| Comply with the license, Use Restrictions and AUP | Content rules in the Terms of Service; H3 scenes screened against them |
| Safeguards: implement, maintain, test, periodically review | Claude screening of every scene (fail closed for H3), Admin → Content Safety: decision log, policy editor, "Test a prompt", "Record periodic review" |
| Do not weaken or allow circumvention | H3 screening cannot be switched off; the policy blocks evasion attempts |
| Reporting mechanism + prompt action | Public /report page, Report button on results, `KINOVI_ABUSE_EMAIL`; admin can remove files and disable accounts |
| Flow-down: bind users, notify them | Terms acceptance required before generating; Terms incorporate Exhibit A and link the license |
| Display "MiniMax H3" | Model is named "MiniMax H3 Naticis" |
| Disclose machine-generated content (AUP item 12) | Terms require users to label AI content; H3 videos carry the disclosure in file metadata |
| Confidentiality, indemnity, $20M revenue line | Business obligations, outside the app |

Still on you: have a lawyer review the Terms draft (`templates/terms.html`), set `KINOVI_ABUSE_EMAIL`, look at the
Content Safety page regularly, and press "Record periodic review" when you do.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Model missing from the list | No endpoint ID set, or the server was not restarted |
| Job fails "cannot run task ref2va" | A Ref2VA scene was sent to the FL2VA endpoint — check the two endpoint IDs are not swapped |
| Job fails with an HTTP error fetching a reference | The worker cannot reach `KINOVI_PUBLIC_ASSET_BASE` |
| Job fails "Could not upload the video back to Cinevora" | The worker cannot reach your server, or the one-time link was already used (a RunPod retry of the same job) — check the server log for `/__upscale_upload__` |
| Scene times out in Cinevora but the job finished later | Queue wait + cold start exceeded `RUNPOD_H3_EXECUTION_TIMEOUT` — raise it, add workers, or keep one active worker |
| First job very slow | Cold start loading the weights; keep one active worker or a longer idle timeout |
| CUDA out of memory | GPU too small for the chosen settings; use B200/H200, or set `SGLANG_ARGS` for memory mode |
