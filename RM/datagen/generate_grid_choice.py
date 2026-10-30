import argparse, os, random
from data import DeepIQGridChoice

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='deepiq_grid', type=str)
parser.add_argument('--num-samples', required=True, type=int)
parser.add_argument('--num-relations', required=True, type=int, choices=[1, 2, 3])
parser.add_argument('--split', default='g-set', choices=['g-set', 's-set'], help='g-set: train shapes, s-set: held-out shapes')
parser.add_argument('--ext', default='train', choices=['train', 'test'])
parser.add_argument('--resolution', default=256, type=int)
parser.add_argument('--seed', default=None, type=int)
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
DeepIQGridChoice.generate(
    args.root, args.num_samples, num_relations=args.num_relations,
    split=args.split, ext=args.ext, resolution=(args.resolution, args.resolution)
)
