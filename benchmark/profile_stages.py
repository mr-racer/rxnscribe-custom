"""Where the time goes in the fork's RxnScribe.predict_images (detection only), per image.

  python benchmark/profile_stages.py --ckpt pix2seq_reaction_full.ckpt --gold <bench>/gold_all.json \
      --images <bench>/images --device cuda --batch_size 16 [--limit 320] [--precision fp16]

Stages: image preparation (resize + pad, one thread), model input (stack, host->device, normalise), backbone,
transformer encoder (+ input projection), decoder, sequence -> boxes + postprocessing. GPU stages are synchronised,
so the numbers are serial costs; in the real pipeline preparation overlaps with the model.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=320)
    ap.add_argument('--precision', default='fp32')
    ap.add_argument('--cuda_graph', default='True')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from rxnscribe import RxnScribe
    from rxnscribe.interface import _prepare_image
    from rxnscribe.data import ReactionImageData, postprocess_reactions_batch

    device = torch.device(args.device)
    with open(args.gold) as f:
        gold = json.load(f)[:args.limit]
    images = [Image.open(os.path.join(args.images, g['file_name'])).convert('RGB') for g in gold]
    model = RxnScribe(args.ckpt, device=device, precision=args.precision, preprocess_threads=0,
                      cuda_graph=args.cuda_graph == 'True')
    tokenizer = model.tokenizer['reaction']
    net = model.model

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    times = defaultdict(float)
    marks = {}

    def pre(name):
        def hook(module, inputs):
            sync()
            marks[name] = time.perf_counter()
        return hook

    def post(name):
        def hook(module, inputs, output):
            sync()
            times[name] += time.perf_counter() - marks[name]
        return hook

    net.backbone.register_forward_pre_hook(pre('backbone'))
    net.backbone.register_forward_hook(post('backbone'))
    net.transformer.encoder.register_forward_pre_hook(pre('encoder'))
    net.transformer.encoder.register_forward_hook(post('encoder'))

    def run(batch, record):
        t0 = time.perf_counter()
        prepared = [_prepare_image(image) for image in batch]
        t1 = time.perf_counter()
        x, refs = model._to_model_input(prepared)
        sync()
        t2 = time.perf_counter()
        with torch.no_grad(), model._autocast():
            seqs, scores = net(x, max_len=tokenizer.max_len)
        sync()
        t3 = time.perf_counter()
        datas = []
        for i, (s, sc) in enumerate(zip(seqs, scores)):
            reactions = tokenizer.sequence_to_data(s.tolist(), sc.tolist(), scale=refs[i]['scale'])
            datas.append(ReactionImageData(predictions=reactions, image=batch[i]))
        postprocess_reactions_batch(datas)
        t4 = time.perf_counter()
        if record:
            times['prepare (1 thread)'] += t1 - t0
            times['model input'] += t2 - t1
            times['model total'] += t3 - t2
            times['postprocess'] += t4 - t3
            times['decode steps'] += sum(len(s) for s in seqs)

    run(images[:args.batch_size], False)   # warm-up / graph capture
    times.clear()
    for idx in range(0, len(images), args.batch_size):
        run(images[idx:idx + args.batch_size], True)
    n = len(images)
    decoder = times['model total'] - times['backbone'] - times['encoder']
    out = {k: round(1000 * v / n, 2) for k, v in times.items() if k != 'decode steps'}
    out['decoder (rest of model)'] = round(1000 * decoder / n, 2)
    out['mean tokens per image'] = round(times['decode steps'] / n, 1)
    out.update({'n': n, 'batch_size': args.batch_size, 'precision': args.precision, 'unit': 'ms per image'})
    print(json.dumps(out))


if __name__ == '__main__':
    main()
