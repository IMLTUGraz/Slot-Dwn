"""
Relational DeepIQ panels: 3x3 matrices whose shape / rotation / zoom / grayscale follow
row, column, diagonal or constant rules.

Requirements: cairosvg, numpy, Pillow, torchvision, tqdm

Usage:
    python data/deepiq.py --root {root} --num-samples 10000 --num-relations 1 --split g-set --ext train
Output: {root}/{folder}/{num_relations}/{split}-{ext}/{i}.png + annotations.json

--split selects the shapes: g-set uses the train shapes, s-set the held-out shapes.
--ext: train or test
"""

import argparse, io, json, os, random, re
import cairosvg
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.v2.functional import affine, InterpolationMode


SHAPES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'deepiq', 'shapes')


def diagonal_indices(side: int = 1, size: int = 3):
    """
    Yields (row_indices, col_indices) for every diagonal in an N x N matrix.

    Args:
        side (int): 1 for (bottom-left to top-right), -1 for (top-left to bottom-right).
        size (int): Dimension N of the N x N matrix.
    """
    if side not in (1, -1):
        raise ValueError("side must be 1 or -1")
    for offset in range(-(size - 1), size):
        rows, cols = [], []
        for r in range(size):
            c = (r if side == 1 else size - 1 - r) + offset
            if 0 <= c < size:
                rows.append(r)
                cols.append(c)
        yield rows, cols


def choice_wo_replace(items, k: int):
    items = list(items).copy()
    random.shuffle(items)
    return items[:k]


class DeepIQ:
    ATTRIBUTES = dict(
        SHAPE = {
            'g-set': [
                'bar-medium', 'bar-narrow', 'rectangle', 'shield', 'square',
                'circle', 'ellipse', 'cup',
                'tee', 'tee-short', 'tee-long',
                'trapezoid',
                'triangle', 'triangle-isosceles', 'triangle-rectangle'
            ],
            's-set': ['club', 'spade', 'diamond', 'heart', 'star']
        },
        ROTATION = [0, 45, 90, 135, 180, 225, 270, 315],
        ZOOM = [0.6, 0.8, 1],
        GRAYSCALE = [0, 72, 109, 182, 255]
    )

    @classmethod
    def matrix(cls, num_relations: int, split: str) -> tuple[np.ndarray, dict[str, str]]:
        """Creates a matrix test with a fixed number of relations."""
        # select attributes that are changed
        relations = ['shape', 'rotation', 'zoom', 'grayscale']
        directions = choice_wo_replace(['row', 'column', 'left-diagonal', 'right-diagonal'], k=num_relations)

        # prepare grid of attributes
        grid = np.full((3, 3, 4), fill_value='_', dtype=np.object_)
        selected = dict(zip(relations, ['constant' for _ in relations]))
        selected |= dict(zip(choice_wo_replace(relations, k=num_relations), directions))

        for relation, direction in selected.items():
            channel = relations.index(relation)
            if relation == 'shape':
                values = choice_wo_replace(cls.ATTRIBUTES[relation.upper()][split], k=3)
            else:
                n = len(cls.ATTRIBUTES[relation.upper()])
                start = random.randint(0, n-3)
                values = cls.ATTRIBUTES[relation.upper()][start:start+3]
                assert len(values) == 3
            if direction == 'row':
                for row in range(3):
                    grid[row, :, channel] = values
            elif direction == 'column':
                for col in range(3):
                    grid[:, col, channel] = values
            elif 'diagonal' in direction:
                for row, col in diagonal_indices(side=1 if direction.startswith('left') else -1, size=3):
                    if len(row) == 1:
                        grid[row, col, channel] = values[0]
                    elif len(row) == 2:
                        grid[row, col, channel] = values[1:]
                    else:
                        grid[row, col, channel] = [values[2], values[0], values[1]]
            elif direction == 'constant':
                grid[:, :, channel] = random.choice(values)

        return grid, selected

    @classmethod
    def renderize(cls, matrix: np.ndarray, resolution: tuple[int, int] = (256, 256)) -> Image.Image:
        H, W = resolution
        img = Image.new('RGBA', (W, H), (255, 255, 255, 255))
        for row in range(3):
            for col in range(3):
                cell = cls.modify(*matrix[row, col].flatten().tolist(), resolution=(H//3, W//3))
                img.paste(cell, (col * W//3, row * H//3), cell)
        return img.convert("LA").convert('L')

    @classmethod
    def modify(cls, shape: str, angle: int, zoom: int, grayscale: int, resolution: tuple[int, int]) -> Image.Image:
        # load the SVG file
        with open(os.path.join(SHAPES_DIR, f'{shape}.svg'), 'r', encoding='utf-8') as f:
            svg_text = f.read()
        hex_color = f"#{grayscale:02x}{grayscale:02x}{grayscale:02x}"
        svg_text = re.sub(r'fill=["\'].*?["\']', f'fill="{hex_color}"', svg_text, count=1)
        png_bytes = cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=resolution[1],
                output_height=resolution[0],
                background_color=None
            )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        img = affine(
            img,
            angle=angle,
            translate=[0, 0],
            scale=zoom,
            shear=0,
            interpolation=InterpolationMode.BILINEAR,
            fill=[0, 0, 0, 0]
        )
        return img

    @classmethod
    def generate(
        cls,
        root: str,
        num_samples: int,
        folder: str,
        num_relations: int,
        split: str,
        ext: str,
        resolution: tuple[int, int] = (256, 256),
        fmt: str = 'png'
    ):
        folder = f'{root}/{folder}/{num_relations}/{split}-{ext}'
        os.makedirs(folder, exist_ok=True)
        annotations = []
        for i in tqdm(range(num_samples), total=num_samples, desc='generate'):
            matrix, selected = cls.matrix(num_relations, split)
            annotations.append(selected)
            cls.renderize(matrix, resolution).save(f'{folder}/{i}.{fmt}')
        annotations = {i: ann for i, ann in enumerate(annotations)}
        with open(f'{folder}/annotations.json', 'w') as writer:
            json.dump(annotations, writer, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=str)
    parser.add_argument('--num-samples', required=True, type=int)
    parser.add_argument('--folder', default='deepiq', type=str)
    parser.add_argument('--num-relations', required=True, type=int, choices=[1, 2, 3])
    parser.add_argument('--split', default='g-set', choices=['g-set', 's-set'], help='g-set: train shapes, s-set: held-out shapes')
    parser.add_argument('--ext', default='train', choices=['train', 'test'])
    parser.add_argument('--resolution', default=256, type=int)
    parser.add_argument('--format', default='png', choices=['png', 'pdf'], help='image file format of the saved panels')
    parser.add_argument('--seed', default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    DeepIQ.generate(
        args.root, args.num_samples, folder=args.folder, num_relations=args.num_relations,
        split=args.split, ext=args.ext, resolution=(args.resolution, args.resolution), fmt=args.format
    )
