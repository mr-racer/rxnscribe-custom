Stand-ins used only by the benchmark to import the *upstream* RxnScribe without its training/download dependencies.
They are appended to the end of `sys.path`, so an installed real package always wins.

* `pycocotools` - upstream `interface.py` imports `dataset.py`, which imports `pycocotools.coco` (training only).
* `huggingface_hub` - `hf_hub_download` returns `$BENCH_MOLSCRIBE_CKPT` instead of downloading.
* `easyocr` - only if EasyOCR is not installed: a Reader whose `readtext` returns [] (OCR is then not timed).
