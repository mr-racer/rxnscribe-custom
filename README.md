# RxnScribe (custom fork)

Reaction diagram parsing (image → reactions with reactant / condition / product boxes, SMILES, text), forked from
[thomas0809/RxnScribe](https://github.com/thomas0809/RxnScribe) at `ad6b1c7`. Same model and checkpoint
(`pix2seq_reaction_full.ckpt`), faster inference, a bug fix, and a way to reuse a MolScribe that is already running.
Molecule recognition uses [molscribe-custom](https://github.com/mr-racer/molscribe-custom).
The original README is kept in [README_upstream.md](README_upstream.md).

## What is different from upstream

| Area | Upstream | This fork |
|---|---|---|
| Decoding | Every step recomputes the key/value projections of the whole image memory (~1.8k positions) in all 6 decoder layers, grows the cache with `torch.cat`, runs the output grammar in Python per sample (host sync each step) | Memory K/V computed once per image, preallocated cache, grammar as two lookup tables on the device, host check every 8 steps, one step captured as a **CUDA graph**. Same tokens and scores |
| Backbone in fp16/bf16 | — (fp32 only) | Frozen BatchNorms folded into the convolutions, `channels_last`: ResNet-50 25 → 10 ms/image |
| Precision | fp32 | `fp32` (bitwise = upstream), `tf32`, `fp16`, `bf16` |
| Preprocessing | PIL + float tensors on the CPU, single thread | same PIL resize in a thread pool while the GPU works; uint8 goes to the GPU and is normalised there (bitwise-identical input) |
| **Bug** | `predict_images` cut molecule/text crops from `input_images[i]`, `i` being the index *inside the batch*: from the second batch on, every image got SMILES/text from crops of another image | fixed |
| MolScribe | its own copy, downloaded from the HF hub at start | any object with `predict_images`: the MolScribe instance of the same process, a checkpoint path, or an HTTP / Celery client of a MolScribe in another container. All crops of a call go in **one** request; a molecule shared by several reactions is read once |
| Start-up | downloads ImageNet ResNet-50 weights, the MolScribe checkpoint, EasyOCR models; imports matplotlib, pycocotools, huggingface_hub | nothing is downloaded (backbone weights are in the checkpoint); MolScribe/EasyOCR created on first use; EasyOCR model folder configurable |
| Dependencies | torch, pandas, matplotlib, pycocotools, pytorch-lightning, transformers, huggingface-hub, MolScribe, easyocr, Pillow==9.5 | core: torch, torchvision, numpy, Pillow, opencv; the rest are extras (`[molscribe]`, `[ocr]`, `[draw]`, `[client]`, `[train]`) |

## Speed (RTX A6000, full `predict_images`, ms per image)

Detection only (reactions and boxes):

| Mode | Upstream | Fork fp32 | Fork fp16 |
|---|---|---|---|
| Batched, `batch_size=16` (1 378 images) | 145 | 52 (×2.8) | **33 (×4.3)** |
| Batched, `batch_size=32` | — | — | 31 |
| One image per call (300 images) | 295 | 102 (×2.9) | 91 (×3.3) |

Full pipeline (`molscribe=True, ocr=True`, MolScribe `swin_base_char_aux_1m.pth`, EasyOCR):

| Mode | Upstream | Fork fp32 | Fork fp16 |
|---|---|---|---|
| Batched, `batch_size=16` | 761 | 160 (×4.8) | **115 (×6.6)** |
| One image per call | 757 | 254 (×3.0) | 239 (×3.2) |

In the full pipeline (fp32) most of the time is MolScribe (≈70 ms/image) and
EasyOCR (≈35 ms/image). Where detection time goes (fp16, batch 16): backbone 10, encoder 8, decoder 11, preprocessing
overlapped with the model.

**Accuracy** (reaction F1, upstream metric, all 1 378 test images; the checkpoint was trained on them, so only the
comparison is meaningful):

| | hard F1 | soft F1 | boxes identical to upstream |
|---|---|---|---|
| Upstream | 0.9196 | 0.9408 | — |
| Fork fp32 | 0.9196 | 0.9408 | 1378 / 1378 |
| Fork tf32 | 0.9196 | 0.9408 | 1346 / 1378 |
| Fork fp16 | 0.9185 | 0.9389 | 922 / 1378 |
| Fork bf16 | 0.9204 | 0.9410 | 155 / 1378 |

fp16/bf16 move some box corners by one coordinate bin; accuracy stays within ±0.2 points.
Reproduce with `benchmark/` (`prepare_data.py`, `run_benchmark.py`, `evaluate.py`, `check_equivalence.py`,
`profile_stages.py`).

## Usage

```python
import torch
from rxnscribe import RxnScribe

model = RxnScribe("pix2seq_reaction_full.ckpt", device=torch.device("cuda"), precision="fp16",
                  molscribe="swin_base_char_aux_1m680k.pth",   # or a MolScribe instance / remote client
                  ocr="/models/easyocr")                       # folder with english_g2.pth, craft_mlt_25k.pth
reactions = model.predict_image_file("scheme.png", molscribe=True, ocr=True)
batch = model.predict_images(list_of_pil_or_rgb_arrays, batch_size=16, molscribe=True, ocr=True)
```

Output format is upstream's: per image a list of reactions, each with `reactants`, `conditions`, `products`, boxes
with `category`, normalised `bbox`, and `smiles`/`molfile` (molecules) or `text` (text boxes).

### Constructor options

| Option | Default | Meaning |
|---|---|---|
| `device` | `cpu` | `torch.device("cuda")` for GPU |
| `precision` | `fp32` | `tf32`, `fp16`, `bf16`. fp32 gives exactly upstream's output |
| `molscribe` | `None` | MolScribe instance, remote client, or checkpoint path. `None`: env `RXNSCRIBE_MOLSCRIBE_CKPT`, else download from the HF hub |
| `molscribe_kwargs` | `{}` | arguments for a MolScribe created from a path, e.g. `{"precision": "fp16"}` |
| `ocr` | `None` | object with `readtext`, or EasyOCR model folder. `None`: env `RXNSCRIBE_EASYOCR_DIR`, else EasyOCR default (downloads) |
| `fast_decoding` | `True` | `False` = upstream decoding loop (A/B checks only) |
| `cuda_graph` | `True` | replay the decoding step as a CUDA graph |
| `fuse_bn`, `channels_last` | auto | on for fp16/bf16 on CUDA, off otherwise |
| `preprocess_threads` | `min(8, cpus)` | `0` disables the prefetch |

`predict_images(images, batch_size=16, molscribe=False, ocr=False, molscribe_batch_size=32)`.

## Choosing a mode

* **Throughput** (a folder of PDFs' figures): `precision="fp16"`, `batch_size=16…32`. Keep the batch size fixed:
  one CUDA graph is captured per distinct batch size (plus one for the last, smaller batch).
* **Latency** (one image per request): `precision="fp32"` or `"tf32"`; fp16 helps little at batch 1.
* Results can differ slightly between batch sizes (cuDNN picks different convolution algorithms), in upstream as well
  (8 of 300 images between batch 1 and 16).

## VRAM (peak allocated by PyTorch; add ≈0.4 GB of CUDA context)

| | fp32 | fp16 |
|---|---|---|
| Detection, batch 1 | 0.7 GB | 0.35 GB |
| Detection, batch 16 | 8.1 GB | 3.2 GB |
| Detection, batch 32 | — | 6.2 GB |
| + MolScribe + EasyOCR, batch 16 | 8.9 GB | 3.8 GB |

Rule of thumb for detection: ≈0.5 GB per image in the batch (fp32), ≈0.2 GB (fp16). MolScribe's own budget is in
its README. To cap memory: `torch.cuda.set_per_process_memory_fraction(...)` and halve `batch_size` on
`OutOfMemoryError`.

## Sharing MolScribe with RxnScribe

RxnScribe needs molecule recognition only through `molscribe.predict_images(crops, batch_size=...)`. Three set-ups,
from most to least efficient; in all of them a MolScribe deployment without RxnScribe works as before.

**1. Same process** (one container, one GPU copy of MolScribe, crops never leave the process). In the MolScribe
FastAPI app:

```python
model = MolScribe(ckpt, device=torch.device("cuda"), precision="fp16")
app.include_router(make_router(model))                    # molscribe.remote: POST /molscribe/predict_batch
try:
    from rxnscribe.serving import attach                  # rxnscribe not installed -> plain MolScribe
except ImportError:
    attach = None
if attach and os.environ.get("RXNSCRIBE_CKPT"):
    attach(app, molscribe=model, ckpt=os.environ["RXNSCRIBE_CKPT"], device=model.device,
           precision="fp16", ocr=os.environ.get("RXNSCRIBE_EASYOCR_DIR"))   # POST /rxnscribe/predict_batch
```

Both models take a lock around their GPU work, so concurrent requests from FastAPI's thread pool are safe.

**2. Separate containers, HTTP** (the model/worker split): the MolScribe container mounts
`molscribe.remote.make_router(model)`; the RxnScribe container uses a client:

```python
from rxnscribe.molscribe_client import HttpMolScribe
model = RxnScribe(ckpt, device=cuda, precision="fp16",
                  molscribe=HttpMolScribe("http://molscribe:8000/molscribe/predict_batch"))
app.include_router(rxnscribe.serving.make_router(model))
```

Images travel as lossless PNG, so results are identical to set-up 1 (checked on 200 images); the cost was
≈9 ms/image (138 vs 129 ms). OCR runs while the MolScribe request is in flight. The RxnScribe container then does not
need MolScribe, timm or RDKit installed.

**3. Celery**: register `molscribe.remote.register_celery_task(celery_app, get_model)` in the MolScribe worker and
use `CeleryMolScribe(celery_app, queue="molscribe")` in RxnScribe. Route the MolScribe task to its own queue/worker,
otherwise a single-slot worker waiting for its own sub-task deadlocks.

## Docker notes

No symlink tricks for caches are needed any more:

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime
RUN pip install "rxnscribe[ocr,client] @ git+https://github.com/mr-racer/rxnscribe-custom.git@main" fastapi uvicorn
# add [molscribe] for set-up 1 (same process)
COPY weights/pix2seq_reaction_full.ckpt /models/
COPY weights/easyocr/ /models/easyocr/           # english_g2.pth, craft_mlt_25k.pth
ENV RXNSCRIBE_CKPT=/models/pix2seq_reaction_full.ckpt RXNSCRIBE_EASYOCR_DIR=/models/easyocr
# set-up 1 only: ENV RXNSCRIBE_MOLSCRIBE_CKPT=/models/swin_base_char_aux_1m680k.pth (if MolScribe is not passed in)
```

The ResNet-50 ImageNet file and the HF-hub cache of the upstream image are no longer used. The repositories are
private: pip needs a token in the URL (`git+https://<token>@github.com/...`) or a copied checkout.

## Files

```
rxnscribe/interface.py             RxnScribe: precision, preprocessing threads, lazy MolScribe/OCR
rxnscribe/pix2seq/static_decoder.py  static decoder, grammar tables, CUDA graph
rxnscribe/molscribe_client.py      HttpMolScribe, CeleryMolScribe, PNG/base64 wire format
rxnscribe/serving.py               FastAPI router, Celery task, attach() next to MolScribe
rxnscribe/data.py                  postprocessing (one MolScribe request per call, shared crops read once)
benchmark/                         data preparation, timing, accuracy, equivalence, profiler
```

Training code and scripts are unchanged from upstream.
