"""Accuracy of saved predictions (upstream metric) and agreement between runs.

  python benchmark/evaluate.py --gold <bench>/gold_cpu100.json results/a.json [results/b.json ...]

For every run: reaction precision/recall/F1 for hard match (all boxes and roles) and soft match (molecules only,
conditions merged into reactants), computed like rxnscribe/evaluate.py. With several runs the first one is the
reference and the others are compared with it image by image (identical boxes and roles; also identical SMILES).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rxnscribe.data import ReactionImageData  # noqa: E402


def metrics(gold, preds, **kwargs):
    gh_sum = gt = ph_sum = pt = 0
    for g, p in zip(gold, preds):
        data = ReactionImageData(g, p)
        gh, ph = data.evaluate(**kwargs)
        gh_sum += sum(gh)
        gt += len(gh)
        ph_sum += sum(ph)
        pt += len(ph)
    precision = ph_sum / max(pt, 1)
    recall = gh_sum / max(gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}


def strip(pred, keep_smiles):
    out = []
    for r in pred:
        rr = {}
        for role in ('reactants', 'conditions', 'products'):
            boxes = []
            for b in r[role]:
                box = {'category': b['category'], 'bbox': [round(x, 6) for x in b['bbox']]}
                if keep_smiles:
                    box['smiles'] = b.get('smiles')
                boxes.append(box)
            rr[role] = boxes
        out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gold', required=True)
    ap.add_argument('--out')
    ap.add_argument('runs', nargs='+')
    args = ap.parse_args()
    with open(args.gold) as f:
        gold = json.load(f)
    by_name = {g['file_name']: g for g in gold}
    report = []
    ref = None
    for path in args.runs:
        with open(path) as f:
            run = json.load(f)
        g = [by_name[n] for n in run['file_names']]
        row = {'name': run['name'], 'n': run['n_images'], 'ms_per_image': round(run['ms_per_image'], 1),
               'hard': metrics(g, run['predictions']),
               'soft': metrics(g, run['predictions'], mol_only=True, merge_condition=True)}
        if ref is None:
            ref = run
        else:
            same_boxes = same_smiles = both = 0
            ref_preds = dict(zip(ref['file_names'], ref['predictions']))
            for name, pred in zip(run['file_names'], run['predictions']):
                if name not in ref_preds:
                    continue
                both += 1
                same_boxes += strip(pred, False) == strip(ref_preds[name], False)
                same_smiles += strip(pred, True) == strip(ref_preds[name], True)
            row['vs_ref'] = {'ref': ref['name'], 'images': both, 'identical_boxes': same_boxes,
                             'identical_with_smiles': same_smiles}
        report.append(row)
        print(json.dumps(row))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
