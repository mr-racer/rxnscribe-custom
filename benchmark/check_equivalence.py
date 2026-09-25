"""Compare the fork's preprocessing and decoding with the upstream code paths on the same images.

  python benchmark/check_equivalence.py --ckpt pix2seq_reaction_full.ckpt --gold <bench>/gold_cpu100.json \
      --images <bench>/images [--limit 20] [--device cuda] [--batch_size 4] [--precision fp32]

1. preprocessing: the model input built by the fork (uint8 -> device -> normalise) vs upstream
   make_transforms('test'), max abs difference and the `scale` reference;
2. decoding: token sequences and scores from the upstream decoding loop vs the static decoder (eager, and CUDA graph
   on CUDA), same encoder output.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--precision', default='fp32')
    args = ap.parse_args()
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shims'))

    import torch
    from PIL import Image
    from rxnscribe import RxnScribe
    from rxnscribe.dataset import make_transforms

    device = torch.device(args.device)
    with open(args.gold) as f:
        gold = json.load(f)[:args.limit]
    images = [Image.open(os.path.join(args.images, g['file_name'])).convert('RGB') for g in gold]
    model = RxnScribe(args.ckpt, device=device, precision=args.precision, preprocess_threads=0)
    transform = make_transforms('test', augment=False, debug=False)

    # 1. preprocessing
    max_diff, scale_mismatch = 0.0, 0
    for image in images:
        ref_x, ref_t = transform(image)
        x, refs = model._to_model_input(model._prepare([image]))
        max_diff = max(max_diff, (x.tensors[0].cpu() - ref_x).abs().max().item())
        scale_mismatch += refs[0]['scale'] != ref_t['scale']
    print(json.dumps({'preprocess_max_abs_diff': max_diff, 'scale_mismatch': scale_mismatch}))

    # 2. decoding
    transformer = model.model.transformer
    max_len = model.tokenizer['reaction'].max_len
    modes = [('reference', False, False), ('static', True, False)]
    if device.type == 'cuda':
        modes.append(('static_graph', True, True))
    outputs = {}
    for name, fast, graph in modes:
        transformer.fast_decoding, transformer.use_cuda_graph = fast, graph
        seqs, scores = [], []
        for idx in range(0, len(images), args.batch_size):
            x, _ = model._to_model_input(model._prepare(images[idx:idx + args.batch_size]))
            with torch.no_grad(), model._autocast():
                s, sc = model.model(x, max_len=max_len)
            seqs += [t.tolist() for t in s]
            scores += [t.float().cpu() for t in sc]
        outputs[name] = (seqs, scores)
    ref_seqs, ref_scores = outputs['reference']
    for name, (seqs, scores) in outputs.items():
        if name == 'reference':
            continue
        same = sum(a == b for a, b in zip(seqs, ref_seqs))
        diffs = [(a - b).abs().max().item() for a, b, s, r in zip(scores, ref_scores, seqs, ref_seqs)
                 if s == r and len(s) > 0]
        print(json.dumps({'mode': name, 'identical_sequences': same, 'n': len(seqs),
                          'max_score_diff': max(diffs) if diffs else 0.0,
                          'empty_reference': sum(len(s) == 0 for s in ref_seqs)}))


if __name__ == '__main__':
    main()
