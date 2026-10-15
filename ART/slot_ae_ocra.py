import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

class SlotAttention(nn.Module):
	def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = dim ** -0.5

		self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
		self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

		self.to_q = nn.Linear(dim, dim)
		self.to_k = nn.Linear(dim, dim)
		self.to_v = nn.Linear(dim, dim)

		self.gru = nn.GRUCell(dim, dim)

		self.fc1 = nn.Linear(dim, dim)
		self.fc2 = nn.Linear(dim, dim)

		self.norm_input  = nn.LayerNorm(dim)
		self.norm_slots  = nn.LayerNorm(dim)
		self.norm_pre_ff = nn.LayerNorm(dim)

		self.register_buffer('fixed_slots', None)

	def freeze(self, num_slots=None, seed=None):
		"""Fix slot initialization."""
		n_s = num_slots if num_slots is not None else self.num_slots
		gen = None
		if seed is not None:
			gen = torch.Generator(device=self.slots_mu.device).manual_seed(seed)
		mu = self.slots_mu.squeeze(0).expand(n_s, -1)
		sigma = self.slots_sigma.squeeze(0).expand(n_s, -1)
		self.fixed_slots = torch.normal(mu, sigma, generator=gen).detach()

	def unfreeze(self):
		self.fixed_slots = None

	def forward(self, inputs, feats,pos,device,num_slots = None):
		b, n, d = inputs.shape
		n_s = num_slots if num_slots is not None else self.num_slots

		if self.fixed_slots is not None:
			slots = self.fixed_slots.unsqueeze(0).expand(b, -1, -1)
		else:
			mu = self.slots_mu.expand(b, n_s, -1)
			sigma = self.slots_sigma.expand(b, n_s, -1)
			slots = torch.normal(mu, sigma)

		inputs = self.norm_input(inputs) 
		pos = pos.expand(b,pos.shape[1],pos.shape[2])
		k, v = self.to_k(inputs), self.to_v(inputs)
		for _ in range(self.iters):
			slots_prev = slots

			slots = self.norm_slots(slots)
			q = self.to_q(slots)

			dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
			attn = dots.softmax(dim=1) + self.eps
			weighted_attn = attn / attn.sum(dim=-1, keepdim=True)
			updates = torch.einsum('bjd,bij->bid', v, weighted_attn)

			slots = self.gru(
				updates.reshape(-1, d),
				slots_prev.reshape(-1, d)
			)

			slots = slots.reshape(b, -1, d)
			slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

		return slots,torch.einsum('bjd,bij->bid', feats, weighted_attn),torch.einsum('bjd,bij->bid', pos, weighted_attn),attn

def build_grid(resolution):
	ranges = [np.linspace(0., 1., num=res) for res in resolution]
	grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
	grid = np.stack(grid, axis=-1)
	grid = np.reshape(grid, [resolution[0], resolution[1], -1])
	grid = np.expand_dims(grid, axis=0)
	grid = grid.astype(np.float32)
	return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
	def __init__(self, hidden_size, resolution):
		"""Builds the soft position embedding layer.
		Args:
		hidden_size: Size of input feature dimension.
		resolution: Tuple of integers specifying width and height of grid.
		"""
		super().__init__()
		self.embedding = nn.Linear(4, hidden_size, bias=True)
		self.grid = build_grid(resolution)

	def forward(self, inputs):
		grid = self.embedding(self.grid)
		return inputs + grid, grid

class Encoder(nn.Module):
	def __init__(self, resolution, hid_dim):
		super().__init__()
		self.conv1 = nn.Conv2d(1, hid_dim, 5, padding = 2)
		self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
		self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
		self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
		self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = F.relu(x)
		feat = x.permute(0,2,3,1)
		y,pos = self.encoder_pos(feat)
		y = torch.flatten(y, 1, 2)
		return y,torch.flatten(feat, 1, 2), torch.flatten(pos, 1, 2)

class Decoder(nn.Module):
	def __init__(self, hid_dim, resolution):
		super().__init__()
		self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
		self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
		self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
		self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
		self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
		self.conv6 = nn.ConvTranspose2d(hid_dim, 2, 3, stride=(1, 1), padding=1)
		self.decoder_initial_size = resolution
		self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
		self.resolution = resolution

	def forward(self, x):
		x,_ = self.decoder_pos(x)
		x = x.permute(0,3,1,2)
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = F.relu(x)
		x = self.conv5(x)
		x = F.relu(x)
		x = self.conv6(x)

		x = x.permute(0,2,3,1)
		return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
	def __init__(self, resolution, num_slots, num_iterations, hid_dim):
		"""Builds the Slot Attention-based auto-encoder.
		Args:
		resolution: Tuple of integers specifying width and height of input image.
		num_slots: Number of slots in Slot Attention.
		num_iterations: Number of iterations in Slot Attention.
		"""
		super().__init__()
		self.hid_dim = hid_dim
		self.resolution = resolution
		self.num_slots = num_slots
		self.num_iterations = num_iterations

		self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
		self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

		self.fc1 = nn.Linear(hid_dim, hid_dim)
		self.fc2 = nn.Linear(hid_dim, hid_dim)

		self.slot_attention = SlotAttention(
			num_slots=self.num_slots,
			dim=hid_dim,
			iters = self.num_iterations,
			eps = 1e-8,
			hidden_dim = 128)

	def freeze(self, seed=None):
		self.slot_attention.freeze(seed=seed)

	def unfreeze(self):
		self.slot_attention.unfreeze()

	def forward(self, image,device):
		# `image` has shape: [batch_size, num_channels, width, height].

		# Convolutional encoder with position embedding.
		x,feat,pos = self.encoder_cnn(image)  # CNN Backbone.
		  

		x = nn.LayerNorm(x.shape[1:]).to(device)(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)  # Feedforward network on set.
		# `x` has shape: [batch_size, width*height, input_size].

		# Slot Attention module.
		slots,feat_slots,pos_slots,attn = self.slot_attention(x,feat,pos,device)
		# `slots` has shape: [batch_size, num_slots, slot_size].

		"""
		Broadcast slot features to a 2D grid and collapse slot dimension.
		"""
		slots_reshaped = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
		slots_reshaped = slots_reshaped.repeat((1, image.shape[2], image.shape[3], 1))
		
		# `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
		
		x = self.decoder_cnn(slots_reshaped)	#shape: [batch_size*num_slots, width, height, num_channels+1].

		# Undo combination of slot and batch dimension; split alpha masks.
		recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([1,1], dim=-1)
		
		# `recons` has shape: [batch_size, num_slots, width, height, num_channels].
		# `masks` has shape: [batch_size, num_slots, width, height, 1].

		# Normalize alpha masks over slots.
		masks = nn.Softmax(dim=1)(masks)
		recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
		recon_combined = recon_combined.permute(0,3,1,2)	# shape: [batch_size, width, height, num_channels].
		
		return recon_combined, recons, masks, feat_slots, attn.reshape(image.shape[0],-1,image.shape[2],image.shape[3],1)