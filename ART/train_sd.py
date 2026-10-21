import os
from slot_ae_ocra import SlotAttentionAutoEncoder
from mh_dwn.dwn import SlotDWN
import time
import torch.optim as optim
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import numpy as np
from util import log, fill_image
import torch.nn as nn
import matplotlib.pyplot as plt
from data import same_diff


class SDdataset(Dataset):
	def __init__(self, root_dir, seq_set, img_size, args):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Resize((img_size,img_size)),
				]
			)

		self.img_size = img_size
		self.all_imgs = []
		for i in range(args.n_shapes):
			img_fname = root_dir +'resizedcropped'+ str(i) + '.npy'
			img = np.load(img_fname)
			self.all_imgs.append(img)

		self.seq_ind = seq_set['seq_ind']
		self.y = seq_set['y']
		self.len = self.seq_ind.shape[0]

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		seq_ind = self.seq_ind[idx]
		y = self.y[idx]

		full_img = 255*np.ones((160,160))

		transx, transy = np.random.randint(-5,5,(2,))
		full_img = fill_image(self.all_imgs,seq_ind[0],full_img,[80+transx,40+transy])
		transx, transy = np.random.randint(-5,5,(2,))
		full_img = fill_image(self.all_imgs,seq_ind[1],full_img,[80+transx,120+transy])

		img = self.transforms(Image.fromarray(full_img.astype(np.uint8)))

		return img, y


def get_feats_and_masks(slot_model, img):
	return slot_model(img, img.device)


def filter_slots(slots, masks, num_obj):
	# B, S, H, W, 1 = masks.shape
	slot_dim = slots.shape[-1]
	masks_thrs = (masks > 0.5).float()
	mask_mean = masks_thrs.flatten(2).mean(dim=-1)
	active_indices = mask_mean.topk(num_obj, dim=-1).indices

	idx = active_indices.unsqueeze(-1).expand(-1, -1, slot_dim)
	active_slots = slots.gather(dim=1, index=idx)

	idx_mask = active_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, masks.shape[-3], masks.shape[-2], 1)
	active_masks = masks.gather(dim=1, index=idx_mask).squeeze(-1)  # (B, num_obj, H, W)
	return active_slots, active_masks, active_indices


def filter_and_sort_slots(slots, masks, num_obj):
	B, S, H, W, _ = masks.shape
	slot_dim = slots.shape[-1]

	active_slots, active_masks, _ = filter_slots(slots, masks, num_obj)

	# Compute column centroid of each slot's mask
	col_grid = torch.linspace(0, 1, W, device=masks.device).view(1, 1, 1, W)
	mask_sum = active_masks.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)  # (B, num_obj, 1, 1)
	col_centroid = (active_masks * col_grid).sum(dim=(-2, -1)) / mask_sum.reshape(B, num_obj)

	sort_idx = col_centroid.argsort(dim=1)  # (B, num_obj): left-to-right

	sort_idx_slots = sort_idx.unsqueeze(-1).expand(-1, -1, slot_dim)
	sorted_slots = active_slots.gather(dim=1, index=sort_idx_slots)

	return sorted_slots, sort_idx


def load_slot_checkpoint(slot_model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn model
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn model with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	slot_model.load_state_dict(model_ckp['slot_model_state_dict'])
	slot_model.eval()
	for p in slot_model.parameters():
		p.requires_grad = False

	return slot_model


def train(args):
	print(args)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

	all_shapes = np.arange(args.n_shapes)
	np.random.shuffle(all_shapes)
	if args.m_holdout > 0:
		train_shapes = all_shapes[args.m_holdout:]
		test_shapes = all_shapes[:args.m_holdout]
	else:
		train_shapes = all_shapes
		test_shapes = all_shapes

	log.info('Generating SD task samples...')
	args, train_set, test_set = same_diff.create_task(args, train_shapes, test_shapes)

	train_data = SDdataset(args.path, train_set, args.img_size, args)
	test_data = SDdataset(args.path, test_set, args.img_size, args)

	train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
							shuffle=True, num_workers=args.num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
							shuffle=False, num_workers=args.num_workers)

	log.info('Building model...')
	slot_model = SlotAttentionAutoEncoder((args.img_size,args.img_size), args.num_slots,args.num_iterations, args.hid_dim)
	slot_model = load_slot_checkpoint(slot_model, args.slot_checkpoint)
	slot_model.to(device)
	slot_model.freeze(seed=args.seed)
	num_obj = 2
	dwn_model = SlotDWN(slot_dim=args.hid_dim, n_slots=num_obj, out_size=2, **args.dwn).to(device)

	ce_criterion = nn.CrossEntropyLoss()
	params = [{'params': dwn_model.parameters()}]
	optimizer = optim.Adam(params, lr=args.learning_rate)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-8)

	all_slots = []
	with torch.no_grad():
		for batch_idx, (img, target) in enumerate(train_dataloader):
			img = img.to(device).float()
			_, recons, masks, feat, attn = get_feats_and_masks(slot_model, img)
			active_slots, *_ = filter_slots(slots=feat, masks=attn, num_obj=num_obj)
			all_slots.append(active_slots)

	dwn_model.fit_binarization(torch.cat(all_slots, dim=0))

	log.info('Training begins...')
	start = time.time()
	best_acc = 0.

	for epoch in range(1, args.num_epochs + 1):
		dwn_model.train()
		all_train_ce_loss = 0.
		all_train_acc = 0.

		for batch_idx, (img, target) in enumerate(train_dataloader):
			img = img.to(device).float()
			target = target.to(device).long()
			dwn_model.clip_luts()

			with torch.no_grad():
				_, recons, masks, feat, attn = get_feats_and_masks(slot_model, img)
				sorted_slots, *_ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)

			score = dwn_model(sorted_slots)
			pred = score.argmax(-1)
			acc = torch.eq(pred, target).float().mean().item() * 100.0
			loss = ce_criterion(score, target)
			ce_loss = loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_idx % args.log_interval == 0:
				log.info('[Epoch: ' + str(epoch) + '] ' +
						  '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_dataloader)) + '] ' +
						  '[CE Loss = ' + '{:.4f}'.format(ce_loss) + '] ' +
						  '[Accuracy = ' + '{:.2f}'.format(acc) + ']')

			all_train_ce_loss += ce_loss
			all_train_acc += acc

		lr_scheduler.step()

		avg_train_loss = all_train_ce_loss / len(train_dataloader)
		avg_train_acc = all_train_acc / len(train_dataloader)
		log.info('[Epoch: ' + str(epoch) + '] ' +
				  '[Avg CE Loss = ' + '{:.4f}'.format(avg_train_loss) + '] ' +
				  '[Avg Accuracy = ' + '{:.2f}'.format(avg_train_acc) + ']')

		if epoch % args.save_interval == 0:
			dwn_model.eval()
			all_test_acc = 0.

			with torch.no_grad():
				for batch_idx, (test_img, test_target) in enumerate(test_dataloader):
					test_img = test_img.to(device).float()
					test_target = test_target.to(device).long()

					_, recons, masks, feat, attn = get_feats_and_masks(slot_model, test_img)
					sorted_slots_test, *_ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)

					score = dwn_model(sorted_slots_test)
					pred = score.argmax(-1)
					acc = torch.eq(pred, test_target).float().mean().item() * 100.0
					all_test_acc += acc

			test_acc = all_test_acc / len(test_dataloader)
			log.info('[Epoch: ' + str(epoch) + '] ' +
					  '[Test Accuracy = ' + '{:.2f}'.format(test_acc) + ']')
			best_acc = max(best_acc, test_acc)

	log.info('Training complete in ' + '{:.2f}'.format(time.time() - start) + 's')
	log.info('Best test accuracy: {}'.format(best_acc))
