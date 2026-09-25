"""Gold annotations for the benchmark: the five upstream test folds (all 1378 images) and a random 100-image subset.

  python benchmark/prepare_data.py --splits data/parse/splits --images <dir with the png files> --out <bench dir>
"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument('--splits', default='data/parse/splits')
ap.add_argument('--images', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--subset', type=int, default=100)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

images = []
for k in range(5):
    with open(os.path.join(args.splits, f'test{k}.json')) as f:
        images += json.load(f)['images']
missing = [im['file_name'] for im in images if not os.path.exists(os.path.join(args.images, im['file_name']))]
assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'gold_all.json'), 'w') as f:
    json.dump(images, f)
subset = sorted(random.Random(args.seed).sample(range(len(images)), args.subset))
with open(os.path.join(args.out, f'gold_cpu{args.subset}.json'), 'w') as f:
    json.dump([images[i] for i in subset], f)
print(len(images), 'images;', args.subset, 'in the subset')
