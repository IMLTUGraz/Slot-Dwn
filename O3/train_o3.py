import os
from util import log
from slot_ae_ocra import SlotAttentionAutoEncoder
from mh_dwn.dwn import SlotDWN
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.optim as optim
from dsprites import OrientationShapes
import time


def get_feats_and_masks(slot_model, img):
	return slot_model(img, img.device)

def filter_slots(slots, masks, num_obj, txt=1):
	# B, S, H, W, 1 = masks.shape
	slot_dim = slots.shape[-1]
	mask_max = masks.flatten(2).max(dim=-1).values
	active_indices = mask_max.topk(num_obj, dim=-1).indices

	idx = active_indices.unsqueeze(-1).expand(-1, -1, slot_dim)
	active_slots = slots.gather(dim=1, index=idx)

	idx_mask = active_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, masks.shape[-3], masks.shape[-2], 1)
	active_masks = masks.gather(dim=1, index=idx_mask).squeeze(-1)  # (B, num_obj, H, W)
	return active_slots, active_masks, active_indices

def filter_and_sort_slots(slots, masks, num_obj,  n_rows=2, n_cols=3):
	"""
	Filter active slots by mask energy, then sort by 2D spatial centroid in
	raster order (top-to-bottom primary, left-to-right secondary).
	"""

	B, S, H, W, _ = masks.shape
	slot_dim = slots.shape[-1]

	active_slots, active_masks, _ = filter_slots(slots, masks, num_obj)

	# Compute 2D centroid (row and column) of each slot's mask
	row_grid = torch.linspace(0, 1, H, device=masks.device).view(1, 1, H, 1)
	col_grid = torch.linspace(0, 1, W, device=masks.device).view(1, 1, 1, W)
	mask_sum = active_masks.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)  # (B, num_obj, 1, 1)

	cy = (active_masks * row_grid).sum(dim=(-2, -1)) / mask_sum.reshape(B, num_obj)
	cx = (active_masks * col_grid).sum(dim=(-2, -1)) / mask_sum.reshape(B, num_obj)

	# rank by row_centroid.
	row_edges = torch.linspace(0, 1, n_rows + 1, device=masks.device)[1:-1]
	row_bucket = torch.bucketize(cy, row_edges).float()

	# Multiply by 10 to guarantee row bucket always dominates cx
	sort_key = row_bucket * 10.0 + cx
	order = sort_key.argsort(dim=1)   # (B, num_obj)
	order_s = order.unsqueeze(-1).expand(-1, -1, active_slots.shape[-1])
	sorted_slots = active_slots.gather(dim=1, index=order_s)

	return sorted_slots, order

def load_slot_checkpoint(slot_model,checkpoint_path):
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


def evaluate(loader, epoch, slot_model, dwn_model, ce_criterion, num_obj, device, split='test'):
	dwn_model.eval()
	all_test_acc = 0.
	all_test_loss = 0.

	with torch.no_grad():
		for batch_idx, (test_img, test_target) in enumerate(loader):
			test_img = test_img.to(device).float()
			test_target = test_target.to(device).long()

			slot_seq = []
			for idx in range(test_img.shape[1]):
				_, recons, masks, feat, attn = get_feats_and_masks(slot_model, test_img[:, idx].unsqueeze(1))
				sorted_slots_test, _ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)
				slot_seq.append(sorted_slots_test)
			slot_panels = torch.stack(slot_seq, dim=1)

			scores = []
			for cand in range(test_img.shape[1]):
				score = dwn_model(slot_panels[:, cand])
				scores.append(score)
			cand_scores = torch.cat(scores, dim=-1)
			pred = cand_scores.argmax(-1)
			acc = torch.eq(pred, test_target).float().mean() * 100.0
			loss = ce_criterion(cand_scores, test_target)
			ce_loss = loss.item()
			all_test_loss += ce_loss
			all_test_acc += acc

	test_loss = all_test_loss / len(loader)
	test_acc = all_test_acc / len(loader)
	print(f"Epoch {epoch}, Average {split} accuracy>>>", test_acc)
	return test_acc, test_loss


def train(args):
	print(args)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

	num_obj = args.num_obj
	resolution = (args.img_size, args.img_size)
	transformation = transforms.Compose([
			transforms.Resize(resolution)
		])

	data_path = os.path.join(args.path, f'{args.data_name}-o3-train.npz')
	test_path = os.path.join(args.path, f'{args.data_name}-o3-test.npz')
	all_data = OrientationShapes.from_path(data_path, resolution=resolution, transforms=transformation)
	test_data = OrientationShapes.from_path(test_path, resolution=resolution, transforms=transformation)

	remainder = len(all_data) - args.train_size - args.val_size
	generator = torch.Generator().manual_seed(args.split_seed) if args.split_seed > 0 else None
	train_data, val_data, _ = random_split(
		all_data, [args.train_size, args.val_size, remainder], generator=generator)

	train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
								   shuffle=True, num_workers=args.num_workers)
	test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,
							shuffle=False, num_workers=args.num_workers)
	val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size,
							shuffle=False, num_workers=args.num_workers)

	print("Number of samples in training set>>", len(train_data))
	print("Number of samples in val set>>", len(val_data))
	print("Number of samples in test set>>", len(test_data))

	log.info('Building model...')
	slot_model = SlotAttentionAutoEncoder((args.img_size,args.img_size), args.num_slots,args.num_iterations, args.hid_dim)
	slot_model = load_slot_checkpoint(slot_model, args.slot_checkpoint)
	slot_model.to(device)
	slot_model.freeze(seed=args.seed)
	dwn_model = SlotDWN(slot_dim=args.hid_dim, n_slots=num_obj, out_size=1, **args.dwn).to(device)

	print('---Model---')
	print(dwn_model)
	ce_criterion = nn.CrossEntropyLoss()

	params = [{'params': dwn_model.parameters()}]
	optimizer = optim.Adam(params, lr=args.learning_rate)
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-8)

	all_slots = []

	with torch.no_grad():
		for batch_idx, (img,target) in enumerate(train_loader):
			img = img.to(device).float()
			batch_slots = []
			for con in range(img.shape[1]):
				_, recons, masks, feat, attn = get_feats_and_masks(slot_model, img[:, con].unsqueeze(1))
				active_slots, _, active_idx = filter_slots(slots=feat, masks=attn, num_obj=num_obj, txt=con)
				batch_slots.append(active_slots)
			all_slots.append(torch.stack(batch_slots, dim=1))

	dwn_model.fit_binarization(torch.cat(all_slots, dim=0))

	log.info('Training begins...')
	start = time.time()
	best_test_acc = 0.
	best_val_loss = float('inf')
	patience_counter = 0

	for epoch in range(1, args.num_epochs + 1):
		dwn_model.train()
		all_train_ce_loss = 0.
		all_train_acc = 0.

		for batch_idx, (img, target) in enumerate(train_loader):
			img = img.to(device).float()
			target = target.to(device).long()
			dwn_model.clip_luts()

			slot_seq = []
			with torch.no_grad():
				for idx in range(img.shape[1]):
					_, recons, masks, feat, attn = get_feats_and_masks(slot_model, img[:, idx].unsqueeze(1))
					sorted_slots, _ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)
					slot_seq.append(sorted_slots)
			slot_panels = torch.stack(slot_seq, dim=1)

			scores = []
			for cand in range(img.shape[1]):
				score = dwn_model(slot_panels[:, cand])
				scores.append(score)
			cand_scores = torch.cat(scores, dim=-1)
			pred = cand_scores.argmax(-1)
			acc = torch.eq(pred, target).float().mean() * 100.0
			loss = ce_criterion(cand_scores, target)
			ce_loss = loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_idx % args.log_interval == 0:
				log.info('[Epoch: ' + str(epoch) + '] ' +
						  '[Batch: ' + str(batch_idx) + ' of ' + str(len(train_loader)) + '] ' +
						  '[CE Loss = ' + '{:.4f}'.format(ce_loss) + '] ' +
						  '[Accuracy = ' + '{:.2f}'.format(acc) + ']')

			all_train_ce_loss += ce_loss
			all_train_acc += acc

		lr_scheduler.step()

		epoch_val_acc, epoch_val_loss = evaluate(val_loader, epoch, slot_model, dwn_model, ce_criterion,
												 num_obj, device, split='val')
		print(f"Epoch {epoch} Average val cross entropy loss>>", epoch_val_loss)

		early_stop = False
		if epoch_val_loss < best_val_loss - args.min_delta:
			best_val_loss = epoch_val_loss
			patience_counter = 0
		else:
			patience_counter += 1
			if patience_counter >= args.patience:
				log.info('Early stopping at epoch ' + str(epoch) +
							': validation loss has not improved by ' + '{:.4f}'.format(args.min_delta) +
							' for ' + str(args.patience) + ' epochs.')
				early_stop = True

		if early_stop or epoch % args.save_interval == 0:
			test_acc, _ = evaluate(test_loader, epoch, slot_model, dwn_model, ce_criterion, num_obj, device)
			best_test_acc = max(best_test_acc, test_acc)
			print("Average test accuracy>>>", best_test_acc)

		if early_stop:
			break

	log.info('Training complete in ' + '{:.2f}'.format(time.time() - start) + 's')
