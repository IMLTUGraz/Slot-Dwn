import torch
from torch.nn.functional import softmax

def layer_mapping(input_size, tuple_lenght, output_size, random=True):
    mapp = lambda : torch.randperm(input_size) if random else torch.arange(input_size)
    num_complete, offset = divmod(tuple_lenght*output_size, input_size)
    if num_complete == 0:
        return mapp()[:offset].reshape(output_size, tuple_lenght).type(torch.int32)
    else:
        complete_mappings = torch.cat([mapp() for i in range(num_complete)])
        offset_mapping = mapp()[:offset]
        return torch.cat([complete_mappings, offset_mapping]).reshape(output_size, tuple_lenght).type(torch.int32)

class LearnableMappingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, tau):
        mapping = weights.argmax(dim=0)
        output = x[:, mapping]
        ctx.save_for_backward(x, weights, tau)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, tau  = ctx.saved_tensors
        weights_grad = ((2*x-1).T @ output_grad)
        input_grad = output_grad @ softmax(weights/tau, dim=0).T
        return input_grad, weights_grad, None, None
    

class LearnableMapping(torch.nn.Module):
    def __init__(self, input_size, output_size, tau=0.001):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(input_size, output_size, dtype=torch.float32), requires_grad=True)
        self.tau = tau

    def forward(self, x):
        return LearnableMappingFunction.apply(x, self.weights, torch.tensor(self.tau))
    
    def extra_repr(self):
        in_size, out_size = self.weights.shape
        return f"input_size={in_size}, output_size={out_size}, tau={self.tau}"


class SubsetLearnableMappingFunction(torch.autograd.Function):
    """
    weights:  (output_size, pool_size, n)   -- scores, one per (node, candidate, slot)
    pool_idx: (output_size, pool_size)      -- global input indices each node may see
    """
    @staticmethod
    def forward(ctx, x, weights, pool_idx, tau):
        local = weights.argmax(dim=1)                       # (O, n) index into the pool
        glob = torch.gather(pool_idx, 1, local)             # (O, n) index into the input
        output = x[:, glob.reshape(-1)]                     # (B, O*n), node-major then slot
        ctx.save_for_backward(x, weights, pool_idx, tau)
        return output

    @staticmethod
    def backward(ctx, output_grad):
        x, weights, pool_idx, tau = ctx.saved_tensors
        B = x.shape[0]
        O, P, n = weights.shape

        g = output_grad.reshape(B, O, n)                    # regroup slots by node
        xp = 2.0 * x[:, pool_idx] - 1.0                     # (B, O, P), the (2x-1) transform

        # (2x-1)^T @ grad, restricted to each node's pool
        weights_grad = torch.einsum('bop,bos->ops', xp, g)  # (O, P, n)

        # grad @ softmax(W/tau)^T, then scattered back
        soft = softmax(weights / tau, dim=1)                # (O, P, n)  softmax over the pool
        contrib = torch.einsum('bos,ops->bop', g, soft)     # (B, O, P)

        input_grad = torch.zeros_like(x)
        input_grad.index_add_(1, pool_idx.reshape(-1), contrib.reshape(B, -1))

        return input_grad, weights_grad, None, None


def build_pools(input_size, output_size, pool_size, mode='random', generator=None):
    """
    mode:
      'random'     each node gets an independent random subset of the input
      'block'      contiguous chunks, nodes cycle through them
      'strided'    evenly spaced bits, offset (s) per node
    """
    assert pool_size <= input_size
    if mode == 'random':
        return torch.stack([torch.randperm(input_size, generator=generator)[:pool_size]
                            for _ in range(output_size)])
    if mode == 'block':
        starts = (torch.arange(output_size) * pool_size) % input_size
        return (starts[:, None] + torch.arange(pool_size)[None, :]) % input_size
    if mode == 'strided':
        stride = max(1, input_size // pool_size)
        offs = torch.arange(output_size)[:, None] % stride
        return (offs + torch.arange(pool_size)[None, :] * stride) % input_size
    raise ValueError(mode)


class SubsetLearnableMapping(torch.nn.Module):
    def __init__(self, input_size, output_size, n, pool_size=100,
                 tau=0.001, mode='random'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n = n
        self.pool_size = pool_size
        self.tau = tau
        self.weights = torch.nn.Parameter(
            torch.rand(output_size, pool_size, n, dtype=torch.float32), requires_grad=True)
        self.register_buffer('pool_idx',
            build_pools(input_size, output_size, pool_size, mode).long())

    def forward(self, x):
        return SubsetLearnableMappingFunction.apply(
            x, self.weights, self.pool_idx, torch.tensor(self.tau, device=x.device))

    def extra_repr(self):
        return (f"input_size={self.input_size}, output_size={self.output_size}, n={self.n}, "
                f"pool_size={self.pool_size}, tau={self.tau}")