import os
import argparse
from slot_ae_ocra import SlotAttentionAutoEncoder
from MH_DWN import SlotDWN
import time
import torch.optim as optim
import torch
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import numpy as np
from util import log
import torch.nn as nn
import matplotlib.pyplot as plt
from data import dist3


plt.rcParams.update({
	'font.family': 'serif',
	'font.serif': ['Times New Roman', 'Liberation Serif', 'Tinos', 'Nimbus Roman', 'STIXGeneral'],
	'mathtext.fontset': 'stix',
	'pdf.fonttype': 42,
	'font.size': 12,
})

DWN_ARG_NAMES = ['dwn_heads', 'dwn_hid', 'dwn_num_layers', 'dwn_levels',
                  'tau', 'n', 'mapping_type', 'lm_tau', 'first_layer_heads',
                  'use_subset_lm', 'subset_lm_mode', 'pool_size']


def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)


def fill_image(all_imgs, idx, full_img, p):
	h = all_imgs[idx].shape[0]
	w = all_imgs[idx].shape[1]
	x = p[0]
	y = p[1]
	if h%2==0 and w%2!=0:
		full_img[x- h//2:x+ h//2 , y- w//2:y+ w//2 +1 ] = all_imgs[idx]
	elif h%2!=0 and w%2==0:
		full_img[x- h//2:x+ h//2 +1 , y- w//2:y+ w//2  ] = all_imgs[idx]
	elif h%2!=0 and w%2!=0:
		full_img[x- h//2:x+ h//2 +1 , y- w//2:y+ w//2 +1 ] = all_imgs[idx]
	else:
		full_img[x- h//2:x+ h//2 , y- w//2:y+ w//2  ] = all_imgs[idx]

	return full_img


class Dist3dataset(Dataset):
	def __init__(self, root_dir, seq_set, img_size, args):
		self.root_dir = root_dir
		self.transforms = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Resize((img_size,img_size))
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
		self.seq_len = 4

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		seq_ind = self.seq_ind[idx]
		y = self.y[idx]

		x_in=[]
		for m in range(self.seq_len):
			full_img = 255*np.ones((160,160))

			transx, transy = np.random.randint(-5,5,(2,))
			full_img = fill_image(self.all_imgs,seq_ind[0],full_img,[40+transx,40+transy])
			transx, transy = np.random.randint(-5,5,(2,))
			full_img = fill_image(self.all_imgs,seq_ind[1],full_img,[40+transx,88+transy])
			transx, transy = np.random.randint(-5,5,(2,))
			full_img = fill_image(self.all_imgs,seq_ind[2],full_img,[40+transx,136+transy])
			transx, transy = np.random.randint(-5,5,(2,))
			full_img = fill_image(self.all_imgs,seq_ind[3],full_img,[120+transx,40+transy])
			transx, transy = np.random.randint(-5,5,(2,))
			full_img = fill_image(self.all_imgs,seq_ind[4],full_img,[120+transx,88+transy])
			transx, transy = np.random.randint(-5,5,(2,))
			full_img = fill_image(self.all_imgs,seq_ind[5+m],full_img,[120+transx,136+transy])

			x_in.append(full_img)

		img = [self.transforms(Image.fromarray(x_in[i].astype(np.uint8))) for i in range(len(x_in))]
		resize_image = torch.stack(img,dim=0)

		return resize_image, y


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


def filter_and_sort_slots(slots, masks, num_obj, n_rows=2, n_cols=3):
	"""
	Filter active slots by mask energy, then sort by 2D spatial centroid in
	raster order (top-to-bottom primary, left-to-right secondary).
	"""
	B, S, H, W, _ = masks.shape

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


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_slots', default=6, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--task', type=str, default='dist3', help="{'same_diff', 'RMTS', 'dist3', 'identity_rules'}")
parser.add_argument('--train_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
parser.add_argument('--test_gen_method', type=str, default='full_space', help="{'full_space', 'subsample'}")
parser.add_argument('--n_shapes', type=int, default=100, help="n = total number of shapes available for training and testing")
parser.add_argument('--m_holdout', type=int, default=95, help="m = number of objects (out of n) withheld during training")
# Training settings
parser.add_argument('--train_set_size', type=int, default=10000)
parser.add_argument('--test_set_size', type=int, default=10000)
parser.add_argument('--train_proportion', type=float, default=0.95)
parser.add_argument('--path', default='imgs/', type=str)
parser.add_argument('--img_size', type=int, default=128)

parser.add_argument('--learning_rate', default=0.00008, type=float)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--save_interval', type=int, default=5)

parser.add_argument('--dwn_heads', default=2, type=int)
parser.add_argument('--dwn_hid', default=1024, type=int)
parser.add_argument('--dwn_num_layers', default=2, type=int)
parser.add_argument('--dwn_levels', default=5, type=int)
parser.add_argument('--tau', default=0.05, type=float)
parser.add_argument('--n', default=4, type=int)
parser.add_argument('--mapping-type', default='paired', choices=['one-to-one', 'paired'])
parser.add_argument('--lm-tau', default=0.001, type=float)
parser.add_argument('--first_layer_heads', default=1, type=int)
parser.add_argument('--use-subset-lm', action='store_true', default=False, help="use sparse version of learnable mapping")
parser.add_argument('--subset-lm-mode', default='strided', choices=['random', 'strided', 'block'], help="mode for SubsetLearnableMapping, used when --use-subset-lm is set")
parser.add_argument('--pool_size', default=None, type=int, help="pool size for SubsetLearnableMapping, used when --use-subset-lm is set")

args = parser.parse_args()
print(args)
args_dict = vars(args)
dwn_kwargs = {k: args_dict[k] for k in DWN_ARG_NAMES}

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

task_gen = __import__(args.task)
log.info('Generating task: ' + args.task + '...')
args, train_set, test_set = task_gen.create_task(args, train_shapes, test_shapes)

train_data = Dist3dataset(args.path, train_set, args.img_size, args)
test_data = Dist3dataset(args.path, test_set, args.img_size, args)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
						shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
						shuffle=False, num_workers=args.num_workers)

log.info('Building model...')
slot_model = SlotAttentionAutoEncoder((args.img_size,args.img_size), args.num_slots,args.num_iterations, args.hid_dim)
slot_model = load_slot_checkpoint(slot_model,'ocra-art.pth.tar')
slot_model.to(device)
slot_model.freeze(seed=args.seed)
num_obj = 6
dwn_model = SlotDWN(slot_dim=args.hid_dim, n_slots=num_obj, out_size=1, **dwn_kwargs).to(device)

ce_criterion = nn.CrossEntropyLoss()
params = [{'params': dwn_model.parameters()}]
optimizer = optim.Adam(params, lr=args.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-8)

all_slots = []
with torch.no_grad():
	for batch_idx, (img, target) in enumerate(train_dataloader):
		img = img.to(device).float()
		batch_slots = []
		for idx in range(img.shape[1]):
			_, recons, masks, feat, attn = get_feats_and_masks(slot_model, img[:, idx])
			active_slots, _, active_idx = filter_slots(slots=feat, masks=attn, num_obj=num_obj)
			batch_slots.append(active_slots)
		all_slots.append(torch.stack(batch_slots, dim=1))

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

		slot_seq = []
		with torch.no_grad():
			for idx in range(img.shape[1]):
				_, recons, masks, feat, attn = get_feats_and_masks(slot_model, img[:, idx])
				sorted_slots, *_ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)
				slot_seq.append(sorted_slots)
		slot_panels = torch.stack(slot_seq, dim=1)

		scores = []
		for cand in range(img.shape[1]):
			score = dwn_model(slot_panels[:, cand])
			scores.append(score)
		cand_scores = torch.cat(scores, dim=-1)
		pred = cand_scores.argmax(-1)
		acc = torch.eq(pred, target).float().mean().item() * 100.0
		loss = ce_criterion(cand_scores, target)
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

				slot_seq = []
				for idx in range(test_img.shape[1]):
					_, recons, masks, feat, attn = get_feats_and_masks(slot_model, test_img[:, idx])
					sorted_slots_test, *_ = filter_and_sort_slots(slots=feat, masks=attn, num_obj=num_obj)
					slot_seq.append(sorted_slots_test)
				slot_panels = torch.stack(slot_seq, dim=1)

				scores = []
				for cand in range(test_img.shape[1]):
					score = dwn_model(slot_panels[:, cand])
					scores.append(score)
				cand_scores = torch.cat(scores, dim=-1)
				pred = cand_scores.argmax(-1)
				acc = torch.eq(pred, test_target).float().mean().item() * 100.0
				all_test_acc += acc

		test_acc = all_test_acc / len(test_dataloader)
		log.info('[Epoch: ' + str(epoch) + '] ' +
				  '[Test Accuracy = ' + '{:.2f}'.format(test_acc) + ']')
		best_acc = max(best_acc, test_acc)

log.info('Training complete in ' + '{:.2f}'.format(time.time() - start) + 's')
log.info('Best test accuracy: {}'.format(best_acc))
