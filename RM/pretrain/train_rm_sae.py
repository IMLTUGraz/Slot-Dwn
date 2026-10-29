import os
import sys
import glob
import argparse
import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
import torchvision.utils as vutils
from torch.utils.data import Dataset

from slot_ae import SlotAutoEncoder
from util import log, check_path


class RMDataset(Dataset):
	def __init__(self, root_dir, img_size, split='train', augment=True):
		self.root = os.path.expandvars(root_dir)
		self.paths = sorted(glob.glob(os.path.join(self.root, '*', split, '*.npz')))
		self.img_size = img_size
		self.train = split == 'train' and augment
		self.resize = transforms.Resize((img_size, img_size))
		self.angles = [0, 90, 180, 270]
		self.len = len(self.paths)  # set accordingly

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		# get the img path using idx and read it
		# assign to panel variable
		with np.load(self.paths[idx]) as data:
			panel = data['panels'][0, ..., 0]  # (1, H, W, 1) -> (H, W)
		image = self.resize(Image.fromarray(panel.astype(np.uint8)))
		brightness_factor = 1
		if self.train:
			hflip_flag = 1 if np.random.rand() < 0.5 else 0
			vflip_flag = 1 if np.random.rand() < 0.5 else 0
			angle = random.sample(self.angles, k=1)[0]
		else:
			hflip_flag = 0
			vflip_flag = 0
			angle = 0

		img = tf.to_tensor(image)
		img = tf.adjust_brightness(img, brightness_factor)
		img = tf.rotate(img, angle=angle)
		if hflip_flag == 1:
			img = tf.hflip(img)
		if vflip_flag == 1:
			img = tf.vflip(img)
		return img


def save_checkpoint(slot_model, cfg, it, ckpt_path):
	check_path(os.path.dirname(ckpt_path))
	torch.save({
		'slot_model_state_dict': slot_model.state_dict(),
		'cfg': cfg,
		'iter': it,
	}, ckpt_path)


def load_slot_model(ckpt_path, device):
	"""Builds the model from the model config saved in the checkpoint, not from the yaml config."""
	ckpt = torch.load(ckpt_path, map_location=device)
	model_cfg = ckpt['cfg']['model']
	slot_model = SlotAutoEncoder(**model_cfg).to(device)
	slot_model.load_state_dict(ckpt['slot_model_state_dict'])
	return slot_model, model_cfg, ckpt['iter']


def visualize_recon(image, recon_combined, recons, masks, save_path=None):
	"""
	Visualizes the input, the combined reconstruction and the per-slot reconstructions.

	Args:
		image: (C, H, W)
		recon_combined: (C, H, W)
		recons: (S, H, W, C)
		masks: (S, H, W, 1)
		save_path: optional output path.
	Returns the figure.
	"""

	image = image.detach().float().cpu().clamp(0., 1.)[0].numpy()
	recon_combined = recon_combined.detach().float().cpu().clamp(0., 1.)[0].numpy()
	slot_recons = (recons * masks + (1. - masks)).detach().float().cpu().clamp(0., 1.)[..., 0].numpy()  # (S, H, W)
	tiles = [image, recon_combined] + list(slot_recons)

	def frame(ax, color='0.35', lw=0.8):
		ax.set_xticks([])
		ax.set_yticks([])
		for spine in ax.spines.values():
			spine.set_edgecolor(color)
			spine.set_linewidth(lw)

	cell, gap, sep = 1.2, 0.12, 0.3
	width = len(tiles) * cell + (len(tiles) - 1) * gap + sep
	fig = plt.figure(figsize=(width, cell))

	x = 0.
	for i, tile in enumerate(tiles):
		ax = fig.add_axes([x / width, 0, cell / width, 1])
		ax.imshow(tile, cmap='gray', vmin=0, vmax=1)
		frame(ax)
		x += cell + gap + (sep if i == 1 else 0.)  # separate the slots from the input and the reconstruction

	if save_path is not None:
		fig.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close(fig)
	return fig



parser = argparse.ArgumentParser()
parser.add_argument('--config', default='rm_slot_ae.yaml', type=str, help='path to yaml config file')
parser.add_argument('--ckpt', default=None, type=str, help='checkpoint to load the model (config and weights) from. Training resumes from its saved iteration')
parser.add_argument('--visualize', action='store_true', help='skip training: load --ckpt and save a matplotlib slot reconstruction figure')
parser.add_argument('--vis-out', default='slot_recon.pdf', type=str, help='output path prefix of the --visualize figures; sample n is saved to <prefix>_<n>.pdf')
parser.add_argument('--vis-n', default=16, type=int, help='number of random samples drawn by --visualize')
args = parser.parse_args()
if args.visualize and args.ckpt is None:
	parser.error('--visualize requires checkpoint via --ckpt')

with open(args.config, 'r') as f:
	cfg = yaml.safe_load(f)
print(cfg)

model_cfg = cfg['model']
data_cfg = cfg['data']
optim_cfg = cfg['optim']
train_cfg = cfg['train']
log_cfg = cfg['logging']

use_cuda = not train_cfg['no_cuda'] and torch.cuda.is_available()
device = torch.device("cuda:" + str(train_cfg['device']) if use_cuda else "cpu")
use_amp = optim_cfg['amp'] and use_cuda

start_iter = 0
if args.ckpt is not None:
	log.info('Loading checkpoint ' + args.ckpt + '...')
	slot_model, model_cfg, start_iter = load_slot_model(args.ckpt, device)
	if model_cfg != cfg['model']:
		log.warning('Saved config of the checkpoint differs from given args' + args.config + ', using the checkpoint\'s')
	cfg['model'] = model_cfg
	log.info('Loaded checkpoint (iter ' + str(start_iter) + ')')
else:
	log.info('Building model...')
	slot_model = SlotAutoEncoder(**model_cfg).to(device)

num_slots = model_cfg['num_slots']

log.info('Loading RM dataset from ' + data_cfg['path'] + '...')

img_size = tuple(model_cfg['img_size'])
img_res = img_size[1]

train_data = RMDataset(data_cfg['path'], img_res, split='train')
test_data = RMDataset(data_cfg['path'], img_res, split='test')

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg['batch_size'],
						shuffle=True, num_workers=data_cfg['num_workers'])

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=cfg['batch_size'],
						shuffle=False, num_workers=data_cfg['num_workers'])

print("Number of samples in training set>>", len(train_data))
print("Number of samples in test set>>", len(test_data))

if args.visualize:
	slot_model.eval()
	out_prefix = os.path.splitext(args.vis_out)[0]
	check_path(os.path.dirname(out_prefix) or '.')
	with torch.no_grad():
		for n, idx in enumerate(random.sample(range(len(test_data)), min(args.vis_n, len(test_data)))):
			img = test_data[idx].unsqueeze(0).to(device).float()
			recon_combined, recons, masks, _, _, _ = slot_model(img)
			out = out_prefix + '_' + str(n) + '.pdf'
			visualize_recon(img[0], recon_combined[0], recons[0], masks[0], save_path=out)
			log.info('Saved ' + out)
	sys.exit()

mse_criterion = nn.MSELoss()


def lr_lambda(step):
	# linear warmup for warmup_steps, then hold at peak LR for hold_steps, then
	# (optionally) exponential decay counted from the end of the hold period
	warmup_steps = optim_cfg['warmup_steps']
	hold_steps = optim_cfg.get('hold_steps', 0)
	if warmup_steps > 0 and step < warmup_steps:
		return (step + 1) / warmup_steps
	if step < warmup_steps + hold_steps:
		return 1.0
	if optim_cfg['use_decay'] and optim_cfg['decay_steps'] > 0:
		decay_step = step - warmup_steps - hold_steps
		return optim_cfg['decay_rate'] ** (decay_step / optim_cfg['decay_steps'])
	return 1.0

log.info('Setting up optimizer...')
optimizer = optim.Adam(slot_model.parameters(), lr=optim_cfg['learning_rate'])

for group in optimizer.param_groups:
	group.setdefault('initial_lr', group['lr'])

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_iter - 1)

log.info('---Model---')
log.info(slot_model)
print('Params', sum(p.numel() for p in slot_model.parameters() if p.requires_grad))
log.info('Training begins...')
start = time.time()

vis_dir = os.path.join(log_cfg['log_dir'], 'vis')
check_path(vis_dir)

best_val_loss = float('inf')
train_iter = iter(train_dataloader)
max_iters = train_cfg['max_iters']

for it in range(start_iter + 1, max_iters + 1):
	slot_model.train()
	try:
		img = next(train_iter)
	except StopIteration:
		train_iter = iter(train_dataloader)	# shuffle again
		img = next(train_iter)

	img = img.to(device).float()

	optimizer.zero_grad()
	with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
		recon_combined, recons, masks, _, _, attn_map = slot_model(img)
		loss = mse_criterion(recon_combined, img)
	loss.backward()
	torch.nn.utils.clip_grad_norm_(slot_model.parameters(), optim_cfg['grad_clip_norm'])
	optimizer.step()
	lr_scheduler.step()

	mse_loss = loss.item()

	if it % train_cfg['log_interval'] == 0:
		log.info('[Iter: ' + str(it) + ' of ' + str(max_iters) + '] ' +
				  '[MSE Loss = ' + '{:.6f}'.format(mse_loss) + '] ' +
				  '[LR = ' + '{:.6f}'.format(optimizer.param_groups[0]['lr']) + ']'
				  )

	if it % train_cfg['visualize_interval'] == 0:
		out = os.path.join(vis_dir, 'train_iter' + str(it) + '.png')
		visualize_recon(img[0], recon_combined[0], recons[0], masks[0], save_path=out)
		log.info('Saved ' + out)

	if it % train_cfg['val_interval'] == 0:
		slot_model.eval()
		all_val_mse_loss = 0.
		vis_batch_idx = np.random.randint(len(test_dataloader))

		with torch.no_grad():
			for batch_idx, val_img in enumerate(test_dataloader):
				val_img = val_img.to(device).float()
				with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
					val_recon_combined, val_recons, val_masks, _, _, val_attn_map = slot_model(val_img)
					val_loss = mse_criterion(val_recon_combined, val_img)
				all_val_mse_loss += val_loss.item()

				if batch_idx == vis_batch_idx:
					out = os.path.join(vis_dir, 'val_iter' + str(it) + '.png')
					visualize_recon(val_img[0], val_recon_combined[0], val_recons[0], val_masks[0], save_path=out)
					log.info('Saved ' + out)

		val_mse_loss = all_val_mse_loss / len(test_dataloader)
		print("Average val MSE loss>>>", val_mse_loss)

		if val_mse_loss < best_val_loss:
			best_val_loss = val_mse_loss
			ckpt_path = os.path.join(log_cfg['ckpt_dir'], log_cfg['model_name'] + '-best.pth.tar')
			save_checkpoint(slot_model, cfg, it, ckpt_path)
			log.info('Saved best checkpoint to ' + ckpt_path + ' (val/mse_loss = ' + '{:.6f}'.format(best_val_loss) + ')')

final_ckpt_path = os.path.join(log_cfg['ckpt_dir'], log_cfg['model_name'] + '-final.pth.tar')
save_checkpoint(slot_model, cfg, max_iters, final_ckpt_path)
log.info('Saved final checkpoint to ' + final_ckpt_path)
log.info('Training complete in ' + '{:.2f}'.format(time.time() - start) + 's')