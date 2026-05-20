#!/usr/bin/env python3
"""
Create selected_seqs_train.json and selected_seqs_test.json for the 6 CO3D
experiment sequences, scanning a DUSt3R-preprocessed CO3D directory.

Run AFTER preprocess_for_training.sh on YSU:
    python3 finetune_blur/make_selected_seqs.py \
        --co3d_processed /mnt/weka/hminasyan/data/co3d_processed
"""

import argparse
import glob
import json
import os

# Two sequences per category (both used for fine-tuning)
SEQUENCES = {
    'teddybear': ['101_11758_21048', '101_11763_21624'],
    'hydrant':   ['106_12648_23157', '106_12653_23216'],
    'cup':       ['12_100_593',      '14_158_900'],
    'bottle':    ['34_1397_4376',    '34_1402_4474'],
    'toybus':    ['111_13154_25988', '104_12348_21852'],
    'toytrain':  ['104_12352_22039', '111_13149_23190'],
}


def get_frame_indices(root, category, seq_id):
    img_dir = os.path.join(root, category, seq_id, 'images')
    jpgs = sorted(glob.glob(os.path.join(img_dir, 'frame*.jpg')))
    return [int(os.path.basename(j)[5:-4]) for j in jpgs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--co3d_processed', required=True,
                   help='DUSt3R-preprocessed CO3D output directory')
    args = p.parse_args()

    all_seqs = {}
    for cat, seq_ids in SEQUENCES.items():
        for seq_id in seq_ids:
            frames = get_frame_indices(args.co3d_processed, cat, seq_id)
            if not frames:
                print(f'WARNING: no preprocessed frames found for {cat}/{seq_id}')
                print(f'  Expected: {args.co3d_processed}/{cat}/{seq_id}/images/frame*.jpg')
            else:
                print(f'  {cat}/{seq_id}: {len(frames)} frames')
                all_seqs.setdefault(cat, {})[seq_id] = frames

    if not all_seqs:
        print('ERROR: no sequences found. Run preprocess_for_training.sh first.')
        return 1

    for split in ('train', 'test'):
        out = os.path.join(args.co3d_processed, f'selected_seqs_{split}.json')
        with open(out, 'w') as f:
            json.dump(all_seqs, f, indent=2)
        print(f'Written: {out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
