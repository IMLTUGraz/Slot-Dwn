import os
import glob
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset

from slot_ae import SlotAutoEncoder
from mh_dwn.dwn import SlotDWN
from util import log, check_path

class RMDataset(Dataset):
	def __init__(self, root_dir, num_relations, img_size):
		root = os.path.expandvars(root_dir)
		self.paths = sorted(glob.glob(os.path.join(root, str(num_relations), '*', '*.npz')))
		self.resize = transforms.Resize((img_size, img_size))

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		"""Returns the 5 candidate panels (5, 1, H, W) in [0, 1] and the index of the correct one."""
		with np.load(self.paths[idx]) as data:
			panels, answer = data['panels'], int(data['answer'])
		img = torch.stack([tf.to_tensor(self.resize(Image.fromarray(p[..., 0]))) for p in panels])
		return img, answer


def load_slot_model(ckpt_path, device, seed=None):
	"""Builds the model from the model config saved in the checkpoint."""
	ckpt = torch.load(ckpt_path, map_location=device)
	model_cfg = ckpt['cfg']['model']
	slot_model = SlotAutoEncoder(**model_cfg).to(device)
	slot_model.load_state_dict(ckpt['slot_model_state_dict'])
	slot_model.eval()
	for p in slot_model.parameters():
		p.requires_grad = False
	slot_model.freeze(seed=seed)
	return slot_model, model_cfg


def get_feats_and_masks(slot_model: SlotAutoEncoder, img:torch.Tensor):
	return slot_model.get(img)


def filter_slots(slots, masks, num_obj):
	slot_dim = slots.shape[-1]
	masks_thrs = (masks > 0.5).float()
	mask_mean = masks_thrs.flatten(2).mean(dim=-1)
	active_indices = mask_mean.topk(num_obj, dim=-1).indices

	idx = active_indices.unsqueeze(-1).expand(-1, -1, slot_dim)
	active_slots = slots.gather(dim=1, index=idx)

	idx_mask = active_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, masks.shape[-2], masks.shape[-1])
	active_masks = masks.gather(dim=1, index=idx_mask)  # (B, num_obj, h, w)
	return active_slots, active_masks, active_indices


def filter_and_sort_slots(slots, masks, num_obj, n_rows=3):
	"""Keeps the num_obj most active slots and orders them row-major by mask centroid."""
	B, S, H, W = masks.shape

	active_slots, active_masks, _ = filter_slots(slots, masks, num_obj)

	row_grid = torch.linspace(0, 1, H, device=masks.device).view(1, 1, H, 1)
	col_grid = torch.linspace(0, 1, W, device=masks.device).view(1, 1, 1, W)
	mask_sum = active_masks.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)

	cy = (active_masks * row_grid).sum(dim=(-2, -1)) / mask_sum.reshape(B, num_obj)
	cx = (active_masks * col_grid).sum(dim=(-2, -1)) / mask_sum.reshape(B, num_obj)

	n_cols = num_obj // n_rows
	by_row = cy.argsort(dim=1, stable=True)[:, :n_rows * n_cols]
	cx_r = cx.gather(1, by_row).view(B, n_rows, n_cols)
	by_col = cx_r.argsort(dim=2, stable=True)
	order = by_row.view(B, n_rows, n_cols).gather(2, by_col).view(B, -1)
	order_s = order.unsqueeze(-1).expand(-1, -1, active_slots.shape[-1])
	sorted_slots = active_slots.gather(dim=1, index=order_s)
	return sorted_slots, order


@torch.no_grad()
def encode_panels(slot_model, img, num_obj):
	"""img: (B, P, C, H, W) -> sorted slot features of every panel, (B, P, num_obj, hid_dim)."""
	slot_seq = []
	for idx in range(img.shape[1]):
		feat, attn = get_feats_and_masks(slot_model, img[:, idx])
		sorted_slots, _ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)
		slot_seq.append(sorted_slots)
	return torch.stack(slot_seq, dim=1)


def score_panels(dwn_model, slot_panels):
	return torch.cat([dwn_model(slot_panels[:, c]) for c in range(slot_panels.shape[1])], dim=-1)


@torch.no_grad()
def evaluate(slot_model, dwn_model, dataloader, num_obj, device):
	dwn_model.eval()
	correct, total = 0, 0
	for img, target in dataloader:
		img = img.to(device).float()
		target = target.to(device).long()
		scores = score_panels(dwn_model, encode_panels(slot_model, img, num_obj))
		correct += torch.eq(scores.argmax(-1), target).sum().item()
		total += target.shape[0]
	return 100.0 * correct / total


def train(args):
	print(args)

	ae_cfg = args.slot_ae
	data_cfg = args.data
	dwn_cfg = args.dwn
	optim_cfg = args.optim
	train_cfg = args.train
	num_relations = args.num_relations
	num_obj = args.num_obj
	save_dir = args.ckpt_dir

	use_cuda = not train_cfg['no_cuda'] and torch.cuda.is_available()
	device = torch.device('cuda:' + str(train_cfg['device']) if use_cuda else 'cpu')

	log.info('Loading slot autoencoder...')
	slot_model, model_cfg = load_slot_model(ae_cfg['ckpt'], device, args.seed)
	img_res = model_cfg['img_size'][1]

	log.info('Loading RM dataset, relations = ' + str(num_relations))
	train_data = RMDataset(data_cfg['train_path'], num_relations, img_res)
	test_data = RMDataset(data_cfg['test_path'], num_relations, img_res)
	train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=optim_cfg['batch_size'],
							shuffle=True, num_workers=data_cfg['num_workers'])
	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=optim_cfg['batch_size'],
							shuffle=False, num_workers=data_cfg['num_workers'])
	print('Number of samples in training set>>', len(train_data))
	print('Number of samples in test set>>', len(test_data))

	log.info('Building model...')
	dwn_model = SlotDWN(slot_dim=slot_model.encoder.hidden_dim, n_slots=num_obj, out_size=1, **dwn_cfg).to(device)
	print('---Model---')
	print(dwn_model)

	ce_criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(dwn_model.parameters(), lr=optim_cfg['learning_rate'])
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=optim_cfg['num_epochs'], eta_min=optim_cfg['min_lr'])


	log.info('Fitting binarization on training slots...')
	all_slots = []
	for img, _ in train_dataloader:
		all_slots.append(encode_panels(slot_model, img.to(device).float(), num_obj))
	dwn_model.fit_binarization(torch.cat(all_slots, dim=0))

	log.info('Training begins...')
	start = time.time()
	best_test_acc = 0.
	num_epochs = optim_cfg['num_epochs']

	for epoch in range(1, num_epochs + 1):
		dwn_model.train()
		all_train_ce_loss = 0.
		all_train_acc = 0.

		for batch_idx, (img, target) in enumerate(train_dataloader):
			img = img.to(device).float()
			target = target.to(device).long()
			dwn_model.clip_luts()

			slot_panels = encode_panels(slot_model, img, num_obj)
			scores = score_panels(dwn_model, slot_panels)
			loss = ce_criterion(scores, target)
			acc = torch.eq(scores.argmax(-1), target).float().mean().item() * 100.0
			ce_loss = loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_idx % train_cfg['log_interval'] == 0:
				log.info('[Epoch: ' + str(epoch) + '] ' +
						 '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_dataloader)) + '] ' +
						 '[CE Loss = ' + '{:.4f}'.format(ce_loss) + '] ' +
						 '[Accuracy = ' + '{:.2f}'.format(acc) + ']')

			all_train_ce_loss += ce_loss
			all_train_acc += acc

		lr_scheduler.step()

		avg_train_loss = all_train_ce_loss / len(train_dataloader)
		avg_train_acc = all_train_acc / len(train_dataloader)
		print('Average training cross entropy loss>>', avg_train_loss)
		print('Average training accuracy>>', avg_train_acc)

		if epoch % train_cfg['eval_interval'] == 0:
			test_acc = evaluate(slot_model, dwn_model, test_dataloader, num_obj, device)
			print('Average test accuracy>>>', test_acc)
			best_test_acc = max(best_test_acc, test_acc)

	log.info('Training complete in ' + '{:.2f}'.format(time.time() - start) + 's')

	check_path(save_dir)
	ckpt_path = os.path.join(save_dir, 'slot_dwn_rm_' + str(num_relations) + '.pth')
	torch.save({'dwn_state_dict': dwn_model.state_dict()}, ckpt_path)
	log.info('Saved ' + ckpt_path)

	if data_cfg.get('transfer_path'):
		log.info('Evaluating on transfer set...')
		transfer_data = RMDataset(data_cfg['transfer_path'], num_relations, img_res)
		transfer_dataloader = torch.utils.data.DataLoader(transfer_data, batch_size=optim_cfg['batch_size'],
								shuffle=False, num_workers=data_cfg['num_workers'])
		print('Number of samples in transfer set>>', len(transfer_data))
		transfer_acc = evaluate(slot_model, dwn_model, transfer_dataloader, num_obj, device)
		print('Transfer test accuracy>>>', transfer_acc)
