"""Time RxnScribe on the benchmark images and save its predictions.

The rxnscribe (and molscribe) packages are imported from --code_root / --molscribe_code_root, so the same script runs
upstream and this fork:

  python benchmark/run_benchmark.py --code_root <rxnscribe checkout> --ckpt pix2seq_reaction_full.ckpt \
      --gold <bench>/gold_cpu100.json --images <bench>/images --out <bench>/results/run --name fork_bs16 \
      [--molscribe --molscribe_code_root <molscribe checkout> --molscribe_ckpt <.pth>] [--ocr] \
      [--batch_size 16 | --per_image] [--opt precision=fp16 --ms_opt precision=fp16]

--opt / --ms_opt are passed to RxnScribe(...) / MolScribe(...) of the fork (ignored for upstream).
"""
import argparse
import inspect
import json
import os
import sys
import time


def parse_opts(items):
    opts = {}
    for item in items or []:
        key, value = item.split('=', 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        if value in ('True', 'False'):
            value = value == 'True'
        opts[key] = value
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code_root', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--per_image', action='store_true', help='latency mode: one predict_images call per image')
    ap.add_argument('--molscribe', action='store_true')
    ap.add_argument('--molscribe_code_root')
    ap.add_argument('--molscribe_ckpt')
    ap.add_argument('--molscribe_batch_size', type=int, default=32)
    ap.add_argument('--ocr', action='store_true')
    ap.add_argument('--ocr_dir')
    ap.add_argument('--opt', nargs='*')
    ap.add_argument('--ms_opt', nargs='*')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--threads', type=int, help='torch.set_num_threads')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(args.code_root))
    if args.molscribe_code_root:
        sys.path.insert(1, os.path.abspath(args.molscribe_code_root))
    sys.path.append(os.path.join(here, 'shims'))
    if args.molscribe_ckpt:
        os.environ['BENCH_MOLSCRIBE_CKPT'] = os.path.abspath(args.molscribe_ckpt)

    import torch
    from PIL import Image
    if args.threads:
        torch.set_num_threads(args.threads)
    import rxnscribe
    from rxnscribe import RxnScribe
    fork = 'precision' in inspect.signature(RxnScribe.__init__).parameters
    device = torch.device(args.device)

    with open(args.gold) as f:
        gold = json.load(f)
    if args.limit:
        gold = gold[:args.limit]
    images = [Image.open(os.path.join(args.images, g['file_name'])).convert('RGB') for g in gold]

    t0 = time.perf_counter()
    if fork:
        kwargs = parse_opts(args.opt)
        if args.molscribe:
            kwargs['molscribe'] = args.molscribe_ckpt
            kwargs['molscribe_kwargs'] = parse_opts(args.ms_opt)
        if args.ocr_dir:
            kwargs['ocr'] = args.ocr_dir
        model = RxnScribe(args.ckpt, device=device, **kwargs)
        # the fork creates MolScribe/EasyOCR on first use; create them now so init time is comparable
        if args.molscribe:
            model.molscribe
        if args.ocr:
            model.ocr_model
    else:
        model = RxnScribe(args.ckpt, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    init_s = time.perf_counter() - t0

    call = dict(molscribe=args.molscribe, ocr=args.ocr)
    if fork:
        call['molscribe_batch_size'] = args.molscribe_batch_size

    def run(batch):
        if args.per_image:
            return [model.predict_images([image], batch_size=1, **call)[0] for image in batch]
        return model.predict_images(batch, batch_size=args.batch_size, **call)

    # warm-up (cudnn autotune, CUDA graph capture, lazy inits); not timed
    warm = images[:1] if args.per_image else images[:args.batch_size]
    run(warm)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    preds = run(images)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    result = {
        'name': args.name,
        'code': 'fork' if fork else 'upstream',
        'code_root': os.path.abspath(args.code_root),
        'rxnscribe_file': rxnscribe.__file__,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch': torch.__version__,
        'threads': torch.get_num_threads(),
        'mode': 'per_image' if args.per_image else f'batch{args.batch_size}',
        'molscribe': args.molscribe, 'ocr': args.ocr,
        'opt': parse_opts(args.opt), 'ms_opt': parse_opts(args.ms_opt),
        'n_images': len(images),
        'init_s': init_s,
        'total_s': total_s,
        'ms_per_image': 1000 * total_s / len(images),
        'images_per_s': len(images) / total_s,
        'peak_mem_gb': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        'file_names': [g['file_name'] for g in gold],
        'predictions': preds,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'{args.name}.json'), 'w') as f:
        json.dump(result, f)
    summary = {k: v for k, v in result.items() if k not in ('predictions', 'file_names', 'code_root', 'rxnscribe_file')}
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
