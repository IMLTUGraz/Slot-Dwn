"""
Multiple-choice DeepIQ grid panels: 
3x3 relational grid as data/deepiq.py (same shapes,
values, rendering and resolution)
5 full composite panels one per candidate completion of the missing bottom-right cell (1 correct + 4 distractors), in shuffled order.
Panels are saved as arrays in a .npz per sample.

Requirements: cairosvg, numpy, Pillow, torchvision, tqdm

Usage:
    python generate_grid_choice.py --root {root} --num-samples 5000 --num-relations 1 --split train

Output: {root}/{folder}/{num_relations}/{split}/{i}.npz
            keys: panels (5,H,W,1) uint8, 
                  relations (0-d object array of the selected dict),
                  answer (int)
            + {root}/{folder}/{num_relations}/{split}/annotations.json
"""

import json, os, random

import numpy as np
from tqdm import tqdm

from .deepiq import DeepIQ, choice_wo_replace

ATTR_INDEX = dict(rotation=1, zoom=2, grayscale=3)

# rotation period in degrees after which a shape looks the same (default 360)
ROTATION_PERIOD = {
    'circle': 45, 'square': 90,
    'bar-medium': 180, 'bar-narrow': 180, 'rectangle': 180, 'ellipse': 180, 'diamond': 180,
}


class DeepIQGridChoice:

    @staticmethod
    def look(panel: tuple) -> tuple:
        """Key that is equal for candidates whose rendered cell looks the same."""
        shape, angle, zoom, grayscale = panel
        return shape, angle % ROTATION_PERIOD.get(shape, 360), zoom, grayscale

    @classmethod
    def make_distractors(cls, grid, split: str) -> tuple[tuple, list[tuple]]:
        """Builds 4 distractor (shape, angle, zoom, grayscale) tuples that all look different
        from the correct answer and from each other.

        3 distractors are copies of actual context cells (not the correct answer) with one of
        rotation/zoom/grayscale changed to a different value. The 4th swaps in a shape that
        appears nowhere in the sample, drawn from the same split's shape pool.
        """
        correct = tuple(grid[2, 2].tolist())
        context = [tuple(grid[r, c].tolist()) for r in range(3) for c in range(3) if (r, c) != (2, 2)]
        seen = {cls.look(correct)}
        distractors = []

        for base in choice_wo_replace(context, k=3):
            for _ in range(20):
                attr = random.choice(list(ATTR_INDEX))
                idx = ATTR_INDEX[attr]
                options = [v for v in DeepIQ.ATTRIBUTES[attr.upper()] if v != base[idx]]
                candidate = list(base)
                candidate[idx] = random.choice(options)
                candidate = tuple(candidate)
                if cls.look(candidate) not in seen:
                    seen.add(cls.look(candidate))
                    distractors.append(candidate)
                    break
            else:
                raise RuntimeError('could not build a unique variant distractor')

        used_shapes = {t[0] for t in context} | {correct[0]}
        shape_pool = [s for s in DeepIQ.ATTRIBUTES['SHAPE'][split] if s not in used_shapes]
        for _ in range(20):
            candidate = (
                random.choice(shape_pool),
                random.choice(DeepIQ.ATTRIBUTES['ROTATION']),
                random.choice(DeepIQ.ATTRIBUTES['ZOOM']),
                random.choice(DeepIQ.ATTRIBUTES['GRAYSCALE']),
            )
            if cls.look(candidate) not in seen:
                seen.add(cls.look(candidate))
                distractors.append(candidate)
                break
        else:
            raise RuntimeError('could not build a unique shape distractor')

        return correct, distractors

    @classmethod
    def generate(
        cls,
        root: str,
        num_samples: int,
        folder: str,
        num_relations: int,
        split: str,
        resolution: tuple[int, int] = (256, 256)
    ):
        out_dir = f'{root}/{folder}/{num_relations}/{split}'
        os.makedirs(out_dir, exist_ok=True)
        annotations = {}
        for i in tqdm(range(num_samples), total=num_samples, desc='generate'):
            grid, selected = DeepIQ.matrix(num_relations, split)
            correct, distractors = cls.make_distractors(grid, split)
            answers = [correct] + distractors
            order = list(range(5))
            random.shuffle(order)

            panels = []
            for slot, orig_idx in enumerate(order):
                variant = grid.copy()
                variant[2, 2] = answers[orig_idx]
                panel = DeepIQ.renderize(variant, resolution)
                panels.append(np.array(panel)[..., None])
            panels = np.stack(panels)

            answer = order.index(0)
            np.savez_compressed(
                f'{out_dir}/{i}.npz',
                panels=panels,
                relations=np.array(selected, dtype=object),
                answer=answer,
            )

            annotations[i] = dict(selected=selected, correct_index=answer)

        with open(f'{out_dir}/annotations.json', 'w') as writer:
            json.dump(annotations, writer, indent=4)
