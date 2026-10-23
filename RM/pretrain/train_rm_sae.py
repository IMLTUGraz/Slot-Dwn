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
import wandb
from torch.utils.tensorboard import SummaryWriter

from slot_ae import SlotAutoEncoder
from util import log, check_path


class RMDataset(Dataset):
	def __init__(self, root_dir, img_size, split='train', augment=True):
		self.root = os.path.expandvars(root_dir)
		self.paths = sorted(glob.glob(os.path.join(self.root, '*', split, '*.png')))
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
		panel = np.array(Image.open(self.paths[idx]))
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


def visualize(image, recon_combined, recons, masks, num_slots, N=8):
	"""
	Tiles input / combined recon / per-slot recon into a single grid image for tensorboard.

	Args:
		image: (B, C, H, W)
		recon_combined: (B, C, H, W)
		recons: (B, S, H, W, C)
		masks: (B, S, H, W, 1)
	"""
	N = min(N, image.shape[0])

	image = image[:N].detach().float().cpu().clamp(0., 1.)
	recon_combined = recon_combined[:N].detach().float().cpu().clamp(0., 1.)

	# per-slot composite: slot recon where attended, white (=1.) background elsewhere
	slot_recons = (recons[:N] * masks[:N] + (1. - masks[:N])).detach().float().cpu().clamp(0., 1.)
	slot_recons = slot_recons.permute(0, 1, 4, 2, 3)  # (N, S, C, H, W)

	# tile
	tiles = torch.cat((
		image[:, None, :, :, :],
		recon_combined[:, None, :, :, :],
		slot_recons,
	), dim=1).flatten(end_dim=1)

	# grid
	grid = vutils.make_grid(tiles, nrow=(1 + 1 + num_slots), pad_value=0.8)

	return grid


def save_checkpoint(slot_model, cfg, it, ckpt_path):
	check_path(os.path.dirname(ckpt_path))
	torch.save({
		'slot_model_state_dict': slot_model.state_dict(),
		'cfg': cfg,
		'iter': it,
	}, ckpt_path)



parser = argparse.ArgumentParser()
parser.add_argument('--config', default='rm_slot_ae.yaml', type=str, help='path to yaml config file')
parser.add_argument('--ckpt', default=None, type=str, help='checkpoint to load model weights from; training resumes from its saved iteration')
parser.add_argument('--visualize', action='store_true', help='skip training: load --ckpt and save a matplotlib slot reconstruction figure')
parser.add_argument('--vis-out', default='slot_recon.pdf', type=str, help='output path prefix of the --visualize figures; sample n is saved to <prefix>_<n>.pdf')
parser.add_argument('--vis-n', default=16, type=int, help='number of random samples drawn by --visualize')
args = parser.parse_args()
if args.visualize and args.ckpt is None:
	parser.error('--visualize requires --ckpt')

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

log.info('Building model...')
slot_model = SlotAutoEncoder(
	**model_cfg
).to(device)

num_slots = model_cfg['num_slots']

start_iter = 0
if args.ckpt is not None:
	ckpt = torch.load(args.ckpt, map_location=device)
	slot_model.load_state_dict(ckpt['slot_model_state_dict'])
	start_iter = ckpt['iter']
	log.info('Loaded checkpoint ' + args.ckpt + ' (iter ' + str(start_iter) + ')')

mse_criterion = nn.MSELoss()

log.info('Setting up optimizer...')
optimizer = optim.Adam(slot_model.parameters(), lr=optim_cfg['learning_rate'])


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


for group in optimizer.param_groups:
	group.setdefault('initial_lr', group['lr'])
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_iter - 1)

run = wandb.init(entity=log_cfg['wandb_entity'], project=log_cfg['wandb_project'], config={**model_cfg, **optim_cfg}, sync_tensorboard=True)
run.define_metric("iter", hidden=True)
run.define_metric("train/*", step_metric="iter")
run.define_metric("val/*", step_metric="iter", summary="min,max")

tb_log_dir = os.path.join(log_cfg['log_dir'], log_cfg['model_name'])
check_path(tb_log_dir)
writer = SummaryWriter(log_dir=tb_log_dir)

log.info('---Model---')
log.info(slot_model)
print('Params', sum(p.numel() for p in slot_model.parameters() if p.requires_grad))
log.info('Training begins...')
start = time.time()

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

		run.log({
			'train/mse_loss': mse_loss,
			'train/lr': optimizer.param_groups[0]['lr'],
			'iter': it,
		})
		writer.add_scalar('train/mse_loss', mse_loss, it)
		writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)

	if it % train_cfg['visualize_interval'] == 0:
		grid = visualize(img, recon_combined, recons, masks, num_slots=num_slots)
		writer.add_image('train/recon', grid, it)

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
					grid = visualize(val_img, val_recon_combined, val_recons, val_masks, num_slots=num_slots)
					writer.add_image('val/recon', grid, it)

		val_mse_loss = all_val_mse_loss / len(test_dataloader)
		run.log({
			'val/mse_loss': val_mse_loss,
			'iter': it,
		})
		writer.add_scalar('val/mse_loss', val_mse_loss, it)
		print("Average val MSE loss>>>", val_mse_loss)

		if val_mse_loss < best_val_loss:
			best_val_loss = val_mse_loss
			ckpt_path = os.path.join(log_cfg['ckpt_dir'], log_cfg['model_name'] + '-best.pth.tar')
			save_checkpoint(slot_model, cfg, it, ckpt_path)
			log.info('Saved best checkpoint to ' + ckpt_path + ' (val/mse_loss = ' + '{:.6f}'.format(best_val_loss) + ')')

final_ckpt_path = os.path.join(log_cfg['ckpt_dir'], log_cfg['model_name'] + '-final.pth.tar')
save_checkpoint(slot_model, cfg, max_iters, final_ckpt_path)
log.info('Saved final checkpoint to ' + final_ckpt_path)
writer.close()
log.info('Training complete in ' + '{:.2f}'.format(time.time() - start) + 's')