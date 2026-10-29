"""
Random (rule-free) DeepIQ panels for slot attention autoencoder pretraining.

Usage:
    python generate_random.py --root {root} --num-samples 150000 --split g-set --ext train
Output: {root}/{folder}/{split}-{ext}/{i}.npz (panels, objects) + annotations.json

--split selects the shapes: g-set uses the train shapes, s-set the held-out shapes.
--ext: train or test
"""
import argparse, json, math, os, random
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.deepiq import DeepIQ

NUM_OBJECTS = 9


def shape_pool(split: str) -> list[str]:
    return DeepIQ.ATTRIBUTES['SHAPE'][split]


def sample_objects(split: str, num_objects: int = NUM_OBJECTS) -> list[dict]:
    shapes = shape_pool(split)
    objects = []
    for _ in range(num_objects):
        objects.append(dict(
            shape=random.choice(shapes),
            rotation=random.choice(DeepIQ.ATTRIBUTES['ROTATION']),
            zoom=random.choice(DeepIQ.ATTRIBUTES['ZOOM']),
            grayscale=random.choice(DeepIQ.ATTRIBUTES['GRAYSCALE'])
        ))
    return objects


def render_shape(shape: str, angle: int, zoom: float, grayscale: int, size: int) -> Image.Image:
    img = DeepIQ.modify(shape, angle, zoom, grayscale, resolution=(size, size))
    return img.crop(img.getchannel('A').getbbox())


def _grid_dims(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def place_free(sizes: list[tuple[int, int]], resolution: tuple[int, int], margin: int, gap: int,
               max_tries: int = 200, max_restarts: int = 50) -> list[tuple[int, int]] | None:
    """One object per shuffled grid cell, jittered inside it, with a `gap`-px overlap check."""
    H, W = resolution
    n = len(sizes)
    rows, cols = _grid_dims(n)
    cell_w, cell_h = (W - 2 * margin) / cols, (H - 2 * margin) / rows
    cells = [(r, c) for r in range(rows) for c in range(cols)][:n]
    order = sorted(range(n), key=lambda i: -sizes[i][0] * sizes[i][1])
    for _ in range(max_restarts):
        random.shuffle(cells)
        boxes = [None] * n
        for i, (row, col) in zip(order, cells):
            w, h = sizes[i]
            x0, y0 = margin + col * cell_w, margin + row * cell_h
            for _ in range(max_tries):
                x = x0 + random.uniform(0, max(0, cell_w - w))
                y = y0 + random.uniform(0, max(0, cell_h - h))
                x, y = min(x, W - margin - w), min(y, H - margin - h)
                if all(x + w + gap <= bx or bx + bw + gap <= x or y + h + gap <= by or by + bh + gap <= y
                       for bx, by, bw, bh in filter(None, boxes)):
                    boxes[i] = (x, y, w, h)
                    break
            else:
                break
        if all(boxes):
            return [(round(x), round(y)) for x, y, _, _ in boxes]
    return None


def place_jitter(sizes: list[tuple[int, int]], resolution: tuple[int, int], margin: int) -> list[tuple[int, int]]:
    """Fallback: same as place_free without the gap check."""
    H, W = resolution
    rows, cols = _grid_dims(len(sizes))
    cell_w, cell_h = (W - 2 * margin) / cols, (H - 2 * margin) / rows
    cells = [(r, c) for r in range(rows) for c in range(cols)][:len(sizes)]
    random.shuffle(cells)
    positions = []
    for (w, h), (row, col) in zip(sizes, cells):
        x0, y0 = margin + col * cell_w, margin + row * cell_h
        x = x0 + random.uniform(0, cell_w - w) if w < cell_w else x0 + (cell_w - w) / 2
        y = y0 + random.uniform(0, cell_h - h) if h < cell_h else y0 + (cell_h - h) / 2
        positions.append((int(x), int(y)))
    return positions


def renderize(
    objects: list[dict],
    resolution: tuple[int, int] = (256, 256),
    obj_scale: float = 1.0,
    margin: int = 4,
    gap: int = 10
) -> Image.Image:
    H, W = resolution
    size = round(min(H, W) // 3 * obj_scale)
    crops = [render_shape(o['shape'], o['rotation'], o['zoom'], o['grayscale'], size) for o in objects]
    sizes = [c.size for c in crops]
    positions = place_free(sizes, resolution, margin, gap)
    if positions is None:
        positions = place_jitter(sizes, resolution, margin)

    img = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    for crop, (x, y) in zip(crops, positions):
        img.paste(crop, (x, y), crop)
    return img.convert("LA").convert('L')


def generate(
    root: str,
    num_samples: int,
    folder: str = 'deepiq_random',
    split: str = 'g-set',
    ext: str = 'train',
    resolution: tuple[int, int] = (256, 256),
    num_objects: int = NUM_OBJECTS,
    **render_kwargs
):
    out_dir = f'{root}/{folder}/{split}-{ext}'
    os.makedirs(out_dir, exist_ok=True)
    annotations = {}
    for i in tqdm(range(num_samples), total=num_samples, desc='generate'):
        objects = sample_objects(split, num_objects)
        panel = renderize(objects, resolution, **render_kwargs)
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
    parser.add_argument('--folder', default='deepiq_random', type=str)
    parser.add_argument('--split', default='g-set', choices=['g-set', 's-set'], help='g-set: train shapes, s-set: held-out shapes')
    parser.add_argument('--ext', default='train', choices=['train', 'test'])
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--num-objects', default=NUM_OBJECTS, type=int, help='shapes per panel')
    parser.add_argument('--obj-scale', default=1.0, type=float, help='shape size relative to the relational panel cell')
    parser.add_argument('--gap', default=10, type=int, help='minimum pixel gap between objects')
    parser.add_argument('--margin', default=4, type=int, help='minimum pixel gap to the canvas border')
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    generate(
        args.root, args.num_samples, folder=args.folder, split=args.split, ext=args.ext,
        resolution=(args.resolution, args.resolution), num_objects=args.num_objects,
        obj_scale=args.obj_scale, gap=args.gap, margin=args.margin
    )
