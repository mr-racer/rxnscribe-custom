"""Clients for a MolScribe that runs in another process or container.

RxnScribe only needs an object with `predict_images(images, batch_size=...) -> [{'smiles', 'molfile'}, ...]`.
Pass one of these as `RxnScribe(..., molscribe=client)` when MolScribe is served elsewhere; in the same process pass the
`molscribe.MolScribe` instance itself.

Wire format (shared with `molscribe.remote` in molscribe-custom): a list of base64-encoded PNG images (RGB), the
answer is the list returned by `MolScribe.predict_images`.
"""
import base64

import cv2
import numpy as np


def encode_image(image) -> str:
    """RGB numpy array (or PIL image) -> base64 PNG. PNG is lossless, so the server sees the exact pixels."""
    image = np.ascontiguousarray(np.asarray(image))
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError(f"cannot encode image of shape {image.shape}")
    return base64.b64encode(buf.tobytes()).decode('ascii')


def decode_image(data: str):
    """base64 PNG -> RGB numpy array."""
    buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class HttpMolScribe:
    """MolScribe behind HTTP: POST {"images": [b64 png], "batch_size": n} -> {"predictions": [...]}.

    The endpoint is the one added by `molscribe.remote.make_router` (default path `/molscribe/predict_batch`).
    """
    remote = True

    def __init__(self, url, timeout=600, session=None, headers=None):
        import requests
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.headers = headers or {}

    def predict_images(self, images, batch_size=32, **kwargs):
        if len(images) == 0:
            return []
        payload = {'images': [encode_image(image) for image in images], 'batch_size': batch_size, **kwargs}
        response = self.session.post(self.url, json=payload, timeout=self.timeout, headers=self.headers)
        response.raise_for_status()
        predictions = response.json()['predictions']
        if len(predictions) != len(images):
            raise RuntimeError(f"MolScribe returned {len(predictions)} predictions for {len(images)} images")
        return predictions


class CeleryMolScribe:
    """MolScribe as a Celery task (registered by `molscribe.remote.register_celery_task`).

    Calling it from inside another Celery task blocks that worker until the MolScribe task finishes; this is
    intended (RxnScribe needs the result) and is why `disable_sync_subtasks=False` is passed. Route the MolScribe task
    to its own queue/worker, or the two tasks can deadlock on a single-slot worker.
    """
    remote = True

    def __init__(self, app, task_name='molscribe.predict_batch', queue=None, timeout=600):
        self.app = app
        self.task_name = task_name
        self.queue = queue
        self.timeout = timeout

    def predict_images(self, images, batch_size=32, **kwargs):
        if len(images) == 0:
            return []
        encoded = [encode_image(image) for image in images]
        options = {'queue': self.queue} if self.queue else {}
        result = self.app.send_task(self.task_name, args=[encoded], kwargs={'batch_size': batch_size, **kwargs},
                                    **options)
        return result.get(timeout=self.timeout, disable_sync_subtasks=False)
