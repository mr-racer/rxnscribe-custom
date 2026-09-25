"""HTTP/Celery entry points for RxnScribe, and mounting it next to an existing MolScribe.

Same process as MolScribe (one GPU copy of MolScribe, crops never leave the process)::

    # in the MolScribe FastAPI app, after `model = MolScribe(...)`
    try:
        from rxnscribe.serving import attach
    except ImportError:          # rxnscribe not installed: MolScribe works exactly as before
        attach = None
    if attach and os.environ.get('RXNSCRIBE_CKPT'):
        attach(app, molscribe=model, ckpt=os.environ['RXNSCRIBE_CKPT'], device=model.device)

Separate container: build `RxnScribe(ckpt, molscribe=HttpMolScribe(url))` and mount `make_router(model)`.
"""
from .molscribe_client import decode_image


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if hasattr(obj, 'item') and getattr(obj, 'ndim', 1) == 0:
        return obj.item()
    return obj


def predict_encoded(model, encoded_images, batch_size=16, molscribe=True, ocr=True):
    """base64 PNG/JPEG images -> RxnScribe predictions (JSON-serialisable)."""
    images = [decode_image(data) for data in encoded_images]
    if not images:
        return []
    return _to_builtin(model.predict_images(images, batch_size=batch_size, molscribe=molscribe, ocr=ocr))


def make_router(model, path='/rxnscribe/predict_batch', max_images=256):
    """FastAPI router: POST {"images": [b64], "batch_size": 16, "molscribe": true, "ocr": true}
    -> {"predictions": [...]} (one list of reactions per image)."""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    class BatchRequest(BaseModel):
        images: list
        batch_size: int = 16
        molscribe: bool = True
        ocr: bool = True

    router = APIRouter()

    @router.post(path)
    def predict_batch(request: BatchRequest):
        if len(request.images) > max_images:
            raise HTTPException(status_code=413, detail=f"at most {max_images} images per request")
        try:
            predictions = predict_encoded(model, request.images, request.batch_size, request.molscribe, request.ocr)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {'predictions': predictions}

    return router


def attach(app, molscribe, ckpt, device=None, path='/rxnscribe/predict_batch', **kwargs):
    """Create RxnScribe on top of an existing MolScribe (instance or remote client) and mount its route on `app`."""
    from .interface import RxnScribe
    model = RxnScribe(ckpt, device=device, molscribe=molscribe, **kwargs)
    app.include_router(make_router(model, path=path))
    return model


def register_celery_task(app, get_model, name='rxnscribe.predict_batch', **task_options):
    """Celery task: `get_model()` returns the worker's RxnScribe."""

    @app.task(name=name, **task_options)
    def predict_batch(encoded_images, batch_size=16, molscribe=True, ocr=True):
        return predict_encoded(get_model(), encoded_images, batch_size, molscribe, ocr)

    return predict_batch
