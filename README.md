# RxnScribe (custom fork)

Picture of a reaction scheme in → a list of reactions out: which boxes are reactants, which are conditions,
which are products, plus the SMILES of every molecule and the text of every label.

A fork of [thomas0809/RxnScribe](https://github.com/thomas0809/RxnScribe) with **the same model and the same
public checkpoint**, 3–7× faster, one upstream bug fixed, and — the point of this fork — able to reuse a
**MolScribe that is already running** instead of loading a second copy of it.

This page is everything needed to put it into a service. The full comparison with upstream, the measurements
and the internals are in [docs/fork-details.md](docs/fork-details.md); the original project's README is
[README_upstream.md](README_upstream.md).

**RxnScribe does not read molecules itself.** It finds the boxes, crops them and hands the crops to MolScribe.
So install [molscribe-custom](https://gitlab.odanchem.org/odanchem/molscribe-custom) first (or connect to the
one already deployed — §3).

---

## 1. Install

Same Python/PyTorch requirements as MolScribe (tested on torch 2.x, CUDA 11.8+). GPU strongly recommended.

```bash
# RxnScribe next to a MolScribe running in another container (HTTP), with text OCR:
pip install "rxnscribe[ocr,client] @ git+https://gitlab.odanchem.org/odanchem/rxnscribe-custom.git@main"

# RxnScribe and MolScribe in one process:
pip install "rxnscribe[ocr,molscribe] @ git+https://gitlab.odanchem.org/odanchem/rxnscribe-custom.git@main"
```

The core install is only `torch`, `torchvision`, `numpy`, `Pillow`, `opencv`. Extras: `[molscribe]` (a local
MolScribe — pulls in rdkit and timm), `[ocr]` (EasyOCR), `[client]` (`requests`, for a remote MolScribe),
`[draw]`, `[train]`.

The repositories are private — pip needs credentials in the URL
(`git+https://oauth2:<token>@gitlab.odanchem.org/...`) or a deploy key.

### Checkpoints

Two or three files, put on a volume:

```bash
wget -P /models https://huggingface.co/yujieq/RxnScribe/resolve/main/pix2seq_reaction_full.ckpt  # ~400 MB
# MolScribe, unless it is reached over HTTP/Celery:
wget -P /models https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_1m680k.pth
```

EasyOCR, if text boxes should be read: let it download itself once on any machine with network
(`python -c "import easyocr; easyocr.Reader(['en'])"`), then copy `~/.EasyOCR/model/english_g2.pth` and
`craft_mlt_25k.pth` (~100 MB) into `/models/easyocr/`. Given a folder, RxnScribe opens EasyOCR with
`download_enabled=False`, so a missing file is an error rather than a silent download.

Nothing is fetched at runtime once these paths are given, so the container works offline. The two model paths
can also come from the environment: `RXNSCRIBE_MOLSCRIBE_CKPT`, `RXNSCRIBE_EASYOCR_DIR`.

### Smoke test

```bash
RXNSCRIBE_MOLSCRIBE_CKPT=/models/swin_base_char_aux_1m680k.pth \
RXNSCRIBE_EASYOCR_DIR=/models/easyocr \
python predict.py --model_path /models/pix2seq_reaction_full.ckpt \
                  --image_path assets/jacs.5b12989-Table-c3.png
```

(`predict.py` hard-codes `cuda` and asks for SMILES and text, hence the two environment variables; without them
it would try to download MolScribe and EasyOCR.)

---

## 2. Use it

```python
import torch
from rxnscribe import RxnScribe

model = RxnScribe(
    "/models/pix2seq_reaction_full.ckpt",
    device=torch.device("cuda"),
    precision="fp16",
    molscribe="/models/swin_base_char_aux_1m680k.pth",  # path, live MolScribe object, or remote client — §3
    ocr="/models/easyocr",                              # folder; omit to skip text recognition
)

reactions = model.predict_image_file("scheme.png", molscribe=True, ocr=True)
batch = model.predict_images(images, batch_size=16, molscribe=True, ocr=True)
```

Output (upstream's format): per image a list of reactions; each reaction has `reactants`, `conditions`,
`products`; each of those is a list of boxes with `category`, a normalised `bbox`, and either `smiles` +
`molfile` (a molecule) or `text` (a label).

`molscribe=False, ocr=False` gives detection only — boxes and their roles, no SMILES. That is 2.5–3.5× faster
and needs no MolScribe at all; useful if only the layout is wanted.

Load the model **once** at start-up. Both models take a lock around their GPU work, so concurrent calls from a
thread pool are safe — they queue.

---

## 3. Connecting to a MolScribe that already exists

RxnScribe only ever calls `molscribe.predict_images(crops, batch_size=...)`. Anything with that method works.
All three layouts below leave an existing MolScribe deployment working exactly as before; pick by how the
system is already split.

### A. One container, one GPU copy of MolScribe — fastest

Crops never leave the process. In the MolScribe FastAPI app, after the model is built:

```python
from rxnscribe.serving import attach

attach(app, molscribe=model, ckpt="/models/pix2seq_reaction_full.ckpt",
       device=model.device, precision="fp16", ocr="/models/easyocr")   # POST /rxnscribe/predict_batch
```

MolScribe's own `/molscribe/predict_batch` keeps working. If `rxnscribe` is not installed, the import fails and
the app stays a plain MolScribe — so this can be guarded and shipped in the same image:

```python
try:
    from rxnscribe.serving import attach
except ImportError:
    attach = None
if attach and os.environ.get("RXNSCRIBE_CKPT"):
    attach(app, molscribe=model, ckpt=os.environ["RXNSCRIBE_CKPT"], device=model.device, precision="fp16")
```

### B. Separate containers, HTTP — when the models are already split

MolScribe's container mounts its router (`molscribe.remote.make_router(model)`, see the MolScribe README);
RxnScribe's container points a client at it:

```python
from rxnscribe.molscribe_client import HttpMolScribe

model = RxnScribe(ckpt, device=torch.device("cuda"), precision="fp16",
                  molscribe=HttpMolScribe("http://molscribe:8000/molscribe/predict_batch"))
app.include_router(rxnscribe.serving.make_router(model))               # POST /rxnscribe/predict_batch
```

Crops travel as lossless PNG, so the results are identical to layout A (verified on 200 images); the cost is
≈9 ms per image (138 vs 129 ms). All crops of one call go in **one** request, and a molecule shared by several
reactions is recognised once. OCR runs while the MolScribe request is in flight. This container does not need
MolScribe, rdkit or timm installed at all.

### C. Celery

`molscribe.remote.register_celery_task(celery_app, get_model)` in the MolScribe worker,
`CeleryMolScribe(celery_app, queue="molscribe")` in RxnScribe. **Route the MolScribe task to its own
queue and worker** — a single-slot worker that waits on its own sub-task deadlocks.

---

## 4. The two knobs that matter: `precision` and `batch_size`

### Latency mode — one scheme per request

```python
model = RxnScribe(ckpt, device=torch.device("cuda"), precision="fp16", molscribe=..., ocr=...)
model.predict_image_file("scheme.png", molscribe=True, ocr=True)      # ≈ 239 ms
```

Use `fp16` (91 ms detection vs 102 ms in fp32); use `fp32` only when the output must match upstream bit for bit.
Note that a MolScribe used at batch 1 should itself run in `fp32`/`tf32`, not fp16 — see its README.

### Throughput mode — a folder of figures

```python
model = RxnScribe(ckpt, device=torch.device("cuda"), precision="fp16", molscribe=..., ocr=...)
model.predict_images(images, batch_size=16, molscribe=True, ocr=True)  # ≈ 115 ms/image
```

Use `fp16` and `batch_size=16…32`, and **keep the batch size fixed**: one CUDA graph is captured per distinct
batch size (plus one for the final, smaller batch), so varying it wastes memory and capture time.

| On an RTX A6000, ms per image | upstream | fork fp32 | fork fp16 |
|---|---|---|---|
| Detection only, batch 16 | 145 | 52 | **33** |
| Detection only, one per call | 295 | 102 | **91** |
| Full pipeline (+MolScribe +OCR), batch 16 | 761 | 160 | **115** |
| Full pipeline, one per call | 757 | 254 | **239** |

In the full pipeline most of the time is no longer RxnScribe: MolScribe ≈70 ms/image and EasyOCR ≈35 ms/image
dominate. If throughput is the goal, tune those two first (MolScribe at `fp16`, large batch).

fp16/bf16 shift some box corners by one coordinate bin; reaction F1 stays within ±0.2 points of fp32
(0.9196 → 0.9185 hard F1).

---

## 5. VRAM

Peak allocated by PyTorch; add ≈0.4 GB for the CUDA context.

| | fp32 | fp16 |
|---|---|---|
| Detection, batch 1 | 0.7 GB | 0.35 GB |
| Detection, batch 16 | 8.1 GB | 3.2 GB |
| Detection, batch 32 | — | 6.2 GB |
| Full pipeline (+MolScribe +EasyOCR), batch 16 | 8.9 GB | 3.8 GB |

Rule of thumb for detection: **≈0.5 GB per image in the batch (fp32), ≈0.2 GB (fp16)**. With MolScribe in the
same process, add its own budget (see its README).

To make a budget a hard limit rather than an estimate:

```python
torch.cuda.set_per_process_memory_fraction(6 * 2**30 / torch.cuda.get_device_properties(0).total_memory)  # 6 GB
```

Catch `torch.cuda.OutOfMemoryError` and halve `batch_size`.

A practical starting point: **fp16, `batch_size=16`, ≈4 GB** covers the full pipeline with MolScribe and OCR in
one process on a 8 GB card.

---

## 6. When something goes wrong

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.OutOfMemoryError` | halve `batch_size` (it is the detection batch that dominates — see §5) |
| Everything works but no SMILES appear | called with `molscribe=False`, or no `molscribe=` was passed to the constructor |
| No `text` on condition boxes | `ocr=` not configured, or `ocr=False` at call time |
| Start-up tries to download from Hugging Face | `molscribe=` / `ocr=` left at `None` — pass explicit paths (or set `RXNSCRIBE_MOLSCRIBE_CKPT`, `RXNSCRIBE_EASYOCR_DIR`) |
| Celery worker hangs forever | layout C with MolScribe on the same queue — give it its own worker |
| First call much slower than the rest | CUDA graph capture; warm up with one dummy image at start-up |
| Results differ slightly between batch sizes | cuDNN picks different convolution algorithms per shape; upstream does this too. Keep the batch size fixed |

---

## 7. Docker

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime
RUN pip install "rxnscribe[ocr,client] @ git+https://oauth2:$TOKEN@gitlab.odanchem.org/odanchem/rxnscribe-custom.git@main" \
                fastapi uvicorn
# add [molscribe] instead of [client] for layout A (one process)
COPY weights/pix2seq_reaction_full.ckpt /models/
COPY weights/easyocr/ /models/easyocr/          # english_g2.pth, craft_mlt_25k.pth
ENV RXNSCRIBE_CKPT=/models/pix2seq_reaction_full.ckpt \
    RXNSCRIBE_EASYOCR_DIR=/models/easyocr
```

`RXNSCRIBE_EASYOCR_DIR` (and `RXNSCRIBE_MOLSCRIBE_CKPT`) are read by the library; `RXNSCRIBE_CKPT` is only a
convention for the app that calls `attach()` in layout A.

Nothing is fetched at runtime, so no cache symlink tricks and no network are needed in the container.
