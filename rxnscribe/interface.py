import argparse
import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from .pix2seq import build_pix2seq_model
from .pix2seq.misc import NestedTensor
from .tokenizer import get_tokenizer
from .data import postprocess_reactions_batch, ReactionImageData

# autocast dtype per precision; 'tf32' keeps fp32 tensors but lets matmuls/convolutions use TF32 tensor cores
PRECISIONS = {'fp32': None, 'tf32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}

INPUT_SIZE = 1333
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# MolScribe checkpoint used when nothing is passed: upstream RxnScribe downloads this one from the HF hub
DEFAULT_MOLSCRIBE_REPO, DEFAULT_MOLSCRIBE_FILE = "yujieq/MolScribe", "swin_base_char_aux_1m.pth"


@contextlib.contextmanager
def _allow_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _prepare_image(image):
    """Resize (longer side -> 1333, PIL bilinear) and pad with white to 1333x1333, kept as uint8 HWC.

    Same pixels as the upstream test transform (LargeScaleJitter(1333, 1, 1) + ToTensor + Normalize), including its
    float32 size arithmetic; the float conversion and normalisation happen later on the model's device.
    """
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_size = torch.tensor(image.size[::-1])                      # (h, w)
    target = torch.tensor([float(INPUT_SIZE)])
    scale = torch.minimum(target / image_size[0], target / image_size[1])
    new_h, new_w = (image_size * scale).round().int().clamp(min=1).tolist()
    resized = np.asarray(TF.resize(image, [new_h, new_w]))
    out = np.full((INPUT_SIZE, INPUT_SIZE, 3), 255, dtype=np.uint8)
    out[:new_h, :new_w] = resized
    return out, {'scale': [new_w / INPUT_SIZE, new_h / INPUT_SIZE]}


class RxnScribe:

    def __init__(self, model_path, device=None, molscribe=None, ocr=None, precision='fp32', fast_decoding=True,
                 cuda_graph=True, preprocess_threads=None, molscribe_kwargs=None):
        """
        RxnScribe Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        :param molscribe: how molecule crops are recognised when `predict_images(..., molscribe=True)`:
            - an object with `predict_images(images, batch_size=...)` (a `molscribe.MolScribe` already loaded in this
              process, or a remote client from `rxnscribe.molscribe_client`) - used as is, nothing else is loaded;
            - a path to a MolScribe checkpoint - loaded on first use on `device`;
            - None - the path in env `RXNSCRIBE_MOLSCRIBE_CKPT`, else the upstream checkpoint from the HF hub.
        :param ocr: text recogniser for `predict_images(..., ocr=True)`: an object with `readtext(image, detail=0)`,
            a directory with the EasyOCR model files (no download), or None (env `RXNSCRIBE_EASYOCR_DIR`, else the
            EasyOCR default, which downloads missing files). Created on first use.
        :param precision: 'fp32', 'tf32', 'fp16' or 'bf16' (the last three only change anything on CUDA).
        :param fast_decoding: static-batch decoder with cached cross-attention; False = upstream decoding loop.
        :param cuda_graph: replay the decoding step as a CUDA graph (CUDA + fast_decoding).
        :param preprocess_threads: threads preparing the next batch while the current one runs; 0 disables.
        :param molscribe_kwargs: extra arguments for a MolScribe created here (e.g. {'precision': 'fp16'}).
        """
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {list(PRECISIONS)}")
        args = self._get_args()
        args.format = 'reaction'
        states = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if device is None:
            device = torch.device('cpu')
        self.device = torch.device(device)
        self.precision = precision
        self.tokenizer = get_tokenizer(args)
        self.model = self.get_model(args, self.tokenizer, self.device, states['state_dict'])
        transformer = self.model.transformer
        transformer.fast_decoding = fast_decoding
        transformer.use_cuda_graph = cuda_graph and self.device.type == 'cuda'
        self._mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        if preprocess_threads is None:
            preprocess_threads = min(8, os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(preprocess_threads) if preprocess_threads > 0 else None
        self._lock = threading.RLock()  # CUDA graphs share static buffers: one model call at a time

        self._molscribe_spec = molscribe
        self._molscribe_kwargs = molscribe_kwargs or {}
        self._molscribe = molscribe if hasattr(molscribe, 'predict_images') else None
        self._ocr_spec = ocr
        self._ocr = ocr if hasattr(ocr, 'readtext') else None

    def _get_args(self):
        parser = argparse.ArgumentParser()
        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')
        # Data
        parser.add_argument('--format', type=str, default='reaction')
        parser.add_argument('--input_size', type=int, default=1333)

        args = parser.parse_args([])
        args.pix2seq = True
        args.pix2seq_ckpt = None
        args.pred_eos = True
        args.pretrained_backbone = False  # every backbone weight is in the checkpoint: no ImageNet download
        return args

    def get_model(self, args, tokenizer, device, model_states):
        def remove_prefix(state_dict):
            return {k.replace('model.', ''): v for k, v in state_dict.items()}

        model = build_pix2seq_model(args, tokenizer[args.format])
        model.load_state_dict(remove_prefix(model_states), strict=False)
        model.to(device)
        model.eval()
        return model

    # ---- molecule and text recognisers (created on first use) ------------------------------------------------------

    @property
    def molscribe(self):
        if self._molscribe is None:
            self._molscribe = self.get_molscribe(self._molscribe_spec)
        return self._molscribe

    @property
    def ocr_model(self):
        if self._ocr is None:
            self._ocr = self.get_ocr_model(self._ocr_spec)
        return self._ocr

    def get_molscribe(self, ckpt_path=None):
        from molscribe import MolScribe
        ckpt_path = ckpt_path or os.environ.get('RXNSCRIBE_MOLSCRIBE_CKPT')
        if not ckpt_path:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(DEFAULT_MOLSCRIBE_REPO, DEFAULT_MOLSCRIBE_FILE)
        return MolScribe(ckpt_path, device=self.device, **self._molscribe_kwargs)

    def get_ocr_model(self, model_dir=None):
        import easyocr
        kwargs = {}
        model_dir = model_dir or os.environ.get('RXNSCRIBE_EASYOCR_DIR')
        if model_dir:
            kwargs = {'model_storage_directory': model_dir, 'download_enabled': False}
        return easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'), **kwargs)

    # ---- model -----------------------------------------------------------------------------------------------------

    def _autocast(self):
        if self.device.type != 'cuda':
            return contextlib.nullcontext()
        if self.precision == 'tf32':
            return _allow_tf32()
        dtype = PRECISIONS[self.precision]
        if dtype is None:
            return contextlib.nullcontext()
        # the autocast weight cache must be off while CUDA graphs are captured
        return torch.autocast('cuda', dtype=dtype, cache_enabled=False)

    def _prepare(self, images):
        """Start preprocessing a batch; PIL resizing releases the GIL, so this overlaps with model work."""
        if self._pool is None:
            return [_prepare_image(image) for image in images]
        return [self._pool.submit(_prepare_image, image) for image in images]

    def _to_model_input(self, prepared):
        prepared = [p.result() if self._pool is not None else p for p in prepared]
        pixels = torch.from_numpy(np.stack([p[0] for p in prepared]))
        if self.device.type == 'cuda':
            pixels = pixels.pin_memory().to(self.device, non_blocking=True)
        # same arithmetic as ToTensor + Normalize, on the device (uint8 crosses the bus, not float32)
        x = pixels.permute(0, 3, 1, 2).float().div(255)
        x = x.sub_(self._mean).div_(self._std)
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=self.device)
        return NestedTensor(x, mask), [p[1] for p in prepared]

    def predict_sequences(self, input_images: List, batch_size=16):
        """Detection only: a list of (reactions, image) before postprocessing."""
        tokenizer = self.tokenizer['reaction']
        batches = [input_images[idx:idx + batch_size] for idx in range(0, len(input_images), batch_size)]
        results = []
        with self._lock:
            pending = self._prepare(batches[0]) if batches else None
            for b in range(len(batches)):
                images, refs = self._to_model_input(pending)
                if b + 1 < len(batches):
                    pending = self._prepare(batches[b + 1])  # prepared by the thread pool while this batch runs
                with torch.no_grad(), self._autocast():
                    pred_seqs, pred_scores = self.model(images, max_len=tokenizer.max_len)
                for i, (seqs, scores) in enumerate(zip(pred_seqs, pred_scores)):
                    reactions = tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=refs[i]['scale'])
                    # upstream passed input_images[i] (i = index inside the batch): wrong image after the first batch
                    results.append((reactions, batches[b][i]))
        return results

    def predict_images(self, input_images: List, batch_size=16, molscribe=False, ocr=False, molscribe_batch_size=32):
        """
        :param input_images: PIL images or RGB numpy arrays.
        :param batch_size: images per forward pass of the reaction model.
        :param molscribe: recognise molecule crops; all crops of the call go to MolScribe in one request.
        :param ocr: read the text crops.
        :param molscribe_batch_size: batch size passed to MolScribe.
        """
        detections = self.predict_sequences(input_images, batch_size=batch_size)
        image_datas = [ReactionImageData(predictions=reactions, image=image) for reactions, image in detections]
        return postprocess_reactions_batch(
            image_datas,
            molscribe=self.molscribe if molscribe else None,
            ocr=self.ocr_model if ocr else None,
            batch_size=molscribe_batch_size)

    def predict_image(self, image, **kwargs):
        predictions = self.predict_images([image], **kwargs)
        return predictions[0]

    def predict_image_files(self, image_files: List, **kwargs):
        input_images = []
        for path in image_files:
            image = PIL.Image.open(path).convert("RGB")
            input_images.append(image)
        return self.predict_images(input_images, **kwargs)

    def predict_image_file(self, image_file: str, **kwargs):
        predictions = self.predict_image_files([image_file], **kwargs)
        return predictions[0]

    def draw_predictions(self, predictions, image=None, image_file=None):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        results = []
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        for r in data.pred_reactions:
            fig, ax = plt.subplots(figsize=(w, h))
            fig.tight_layout()
            canvas = FigureCanvasAgg(fig)
            ax.imshow(data.image)
            ax.axis('off')
            r.draw(ax)
            canvas.draw()
            buf = canvas.buffer_rgba()
            results.append(np.asarray(buf))
            plt.close(fig)
        return results

    def draw_predictions_combined(self, predictions, image=None, image_file=None):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        assert image or image_file
        data = ReactionImageData(predictions=predictions, image=image, image_file=image_file)
        h, w = np.array([data.height, data.width]) * 10 / max(data.height, data.width)
        n = len(data.pred_reactions)
        fig, axes = plt.subplots(n, 1, figsize=(w, h * n))
        if n == 1:
            axes = [axes]
        fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.99))
        canvas = FigureCanvasAgg(fig)
        for i, r in enumerate(data.pred_reactions):
            ax = axes[i]
            ax.imshow(data.image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'reaction # {i}', fontdict={'fontweight': 'bold', 'fontsize': 14})
            r.draw(ax)
        canvas.draw()
        buf = canvas.buffer_rgba()
        result_image = np.asarray(buf)
        plt.close(fig)
        return result_image
