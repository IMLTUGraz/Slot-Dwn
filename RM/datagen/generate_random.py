"""
Random (rule-free) DeepIQ panels for slot attention autoencoder pretraining.

Same fixed 3x3 grid layout as generate_grid_choice.py (rendered with DeepIQ.renderize),
but every cell gets an independently random shape, rotation, zoom and grayscale.

Usage:
    python generate_random.py --root {root} --num-samples 150000 --split g-set --ext train
Output: {root}/{split}/{ext}/{i}.npz (panels, objects) + annotations.json

--split selects the shapes: g-set uses the train shapes, s-set the held-out shapes.
--ext: train or test
"""
import argparse, json, os, random
import numpy as np
from tqdm import tqdm

from data.deepiq import DeepIQ

KEYS = ('shape', 'rotation', 'zoom', 'grayscale')


def sample_grid(split: str) -> np.ndarray:
    """3x3 grid of (shape, rotation, zoom, grayscale), each cell sampled independently."""
    grid = np.empty((3, 3, 4), dtype=np.object_)
    for row in range(3):
        for col in range(3):
            grid[row, col] = [
                random.choice(DeepIQ.ATTRIBUTES['SHAPE'][split]),
                random.choice(DeepIQ.ATTRIBUTES['ROTATION']),
                random.choice(DeepIQ.ATTRIBUTES['ZOOM']),
                random.choice(DeepIQ.ATTRIBUTES['GRAYSCALE']),
            ]
    return grid


def generate(
    root: str,
    num_samples: int,
    split: str = 'g-set',
    ext: str = 'train',
    resolution: tuple[int, int] = (256, 256)
):
    out_dir = f'{root}/{split}/{ext}'
    os.makedirs(out_dir, exist_ok=True)
    annotations = {}
    for i in tqdm(range(num_samples), total=num_samples, desc='generate'):
        grid = sample_grid(split)
        objects = [dict(zip(KEYS, grid[r, c].tolist())) for r in range(3) for c in range(3)]
        panel = DeepIQ.renderize(grid, resolution)
        np.savez_compressed(
            f'{out_dir}/{i}.npz',
            panels=np.array(panel)[None, ..., None],
            objects=np.array(objects, dtype=object),
        )
        annotations[i] = dict(objects=objects)

    with open(f'{out_dir}/annotations.json', 'w') as writer:
        json.dump(annotations, writer, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=str)
    parser.add_argument('--num-samples', required=True, type=int)
    parser.add_argument('--split', default='g-set', choices=['g-set', 's-set'], help='g-set: train shapes, s-set: held-out shapes')
    parser.add_argument('--ext', default='train', choices=['train', 'test'])
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    generate(
        args.root, args.num_samples, split=args.split, ext=args.ext,
        resolution=(args.resolution, args.resolution)
    )
