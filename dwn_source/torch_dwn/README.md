## The implementation of MH-DWN is built upon *Differentiable Weightless Neural Networks* (https://github.com/alanbacellar/DWN/tree/main)

This `torch_dwn` package extends the original `LUTLayer` with a **sparse (subset) learnable mapping** (`mapping='subset-learnable'`), utilities to **freeze** a trained mapping into a fixed one, and **binary LUT export**.

---

## Installation

Run from the `dwn_source/` directory:

```bash
cd dwn_source
pip install . --no-build-isolation
```

`torch_dwn/build_dwn.sh` is an example SLURM build script; submit it from `dwn_source/`.

Requirements: CUDA and PyTorch (matching the CUDA version). The LUT forward/backward pass is a CUDA kernel (`efd_cuda`), so layers and inputs must be on a GPU.

---

## Quick Start

### Example

An MNIST example using the `LUTLayer` with the sparse learnable mapping in the first layer.

```python
import torch
from torch import nn
from torchvision import datasets
import torch_dwn as dwn

device = 'cuda:0'

def load_mnist(root='./data'):
    train = datasets.MNIST(root=root, train=True, download=True)
    test = datasets.MNIST(root=root, train=False, download=True)
    prep = lambda d: d.data.float().div(255).flatten(start_dim=1)
    return prep(train), train.targets, prep(test), test.targets

x_train, y_train, x_test, y_test = load_mnist()

thermometer = dwn.DistributiveThermometer(num_bits=3).fit(x=x_train)
x_train = thermometer.binarize(x=x_train).flatten(start_dim=1).to(device)
x_test = thermometer.binarize(x=x_test).flatten(start_dim=1).to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

model = nn.Sequential(
    dwn.LUTLayer(
                input_size=x_train.size(1), 
                output_size=2000, 
                n=4,
                mapping='subset-learnable', 
                subset_lm_mode='random', 
                pool_size=512),
    dwn.LUTLayer(
                input_size=2000, 
                output_size=1000, 
                n=4),
    dwn.GroupSum(
                k=10, 
                tau=1/0.3)
).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-8)

def clip_luts(model):   # LUT values are clipped between [-1, 1]
    for m in model.modules():
        if isinstance(m, dwn.LUTLayer):
            m.clip_weights()

def train(model, x, y, optimizer, scheduler, batch_size=32, epochs=30):
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n=x.size(0), device=x.device)
        for i in range(0, x.size(0), batch_size):
            idx = perm[i:i + batch_size]
            loss = nn.functional.cross_entropy(input=model(x[idx]), target=y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clip_luts(model=model)
        scheduler.step()
        print(f'epoch {epoch + 1}: last batch loss {loss.item():.4f}')

@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    correct = sum((model(x[i:i + batch_size]).argmax(dim=-1) == y[i:i + batch_size]).sum().item()
                  for i in range(0, x.size(0), batch_size))
    print(f'accuracy: {correct / x.size(0):.4f}')

train(model=model, x=x_train, y=y_train, optimizer=optimizer, scheduler=scheduler,
      batch_size=32, epochs=30)
evaluate(model=model, x=x_test, y=y_test)

dwn.freeze(model=model)
evaluate(model=model, x=x_test, y=y_test)
```

---

## Quick Docs

### `DistributiveThermometer`

```python
DistributiveThermometer(num_bits, feature_wise=True)
```

| Argument | Description |
|----------|-------------|
| `num_bits` | Number of bits per feature in the encoding. |
| `feature_wise` | Whether to fit the encoding feature-wise or globally. |

- `.fit(x)`: Fit the encoding using the training data.
- `.binarize(x)`: Binarize new data. **Returns**: Encoded input with shape: (*x.shape, num_bits).

### `LUTLayer`

Defaults for the LUTLayer implementation are shown below:
```python
LUTLayer(input_size, output_size, n, mapping='random', alpha=None, beta=None, ste=True,
         clamp_luts=True, lm_tau=0.001, subset_lm_mode='random', pool_size=None)
```

| Argument | Description |
|----------|-------------|
| `input_size` | Input size for the layer. |
| `output_size` | Output size (number of LUTs). |
| `n` | Number of inputs each LUT receives, i.e., rank of LUT (LUT size is 2^n). |
| `mapping` | `'random'`, `'arange'`, `'learnable'`, `'subset-learnable'`, or a fixed int32 tensor. |
| `alpha` | Linear scalar for EFD. Defaults to `0.5 * 0.75**(n-1)`. |
| `beta` | Exponential scalar for EFD. Defaults to `0.25/0.75`. |
| `ste` | Whether to use Straight-Through Estimator (STE) for binarization. |
| `clamp_luts` | Enables LUT clamping to `[-1, 1]` when `clip_weights()` is called in training. |
| `lm_tau` | Softmax temperature for the backward pass of the learnable mappings. |
| `subset_lm_mode` | Pool construction for `'subset-learnable'`: `'random'`, `'block'` or `'strided'`. |
| `pool_size` | Candidates per LUT for `'subset-learnable'`. Defaults to `max(n, input_size // 8)`. |

#### Mapping options

| `mapping` | Trainable | Description |
|-----------|-----------|-------------|
| `'random'` | no | Each LUT is wired to `n` random inputs.|
| `'arange'` | no | Sequential wiring of inputs e.g., (`0, 1, 2, ...`). |
| `'learnable'` | yes | Every LUT input can pick from **all** `input_size` bits. Parameters: `input_size * output_size * n`. |
| `'subset-learnable'` | yes | Every LUT input picks from a **pool of `pool_size` bits** assigned to that LUT. Parameters: `output_size * pool_size * n`. |
| `torch.Tensor` | no | A user-defined fixed mapping: an `int32` tensor of shape `(output_size, n)`. |

In the learnable mappings, the forward pass uses a hard `argmax` selection and the backward pass uses a softmax with temperature `lm_tau`.

#### Sparse (subset) learnable mapping

The full `'learnable'` mapping stores a score for every (input bit, LUT input) pair, which grows quickly with `input_size`. `'subset-learnable'` restricts each LUT to its own pool of `pool_size` candidate bits and learns which `n` of them to use, so memory and compute scale with `pool_size` rather than `input_size`.

`subset_lm_mode` controls how the pools are built:

| `subset_lm_mode` | Pool of LUT `i` |
|------------------|-----------------|
| `'random'` | An independent random subset of the input bits. |
| `'block'` | A contiguous chunk of `pool_size` bits, consecutive LUTs move to the next chunk (cycling over the input). |
| `'strided'` | Evenly spaced bits (stride `input_size // pool_size`), shifted by a per-LUT offset. |

`pool_size` must be `<= input_size`. If it is not given it defaults to `max(n, input_size // 8)`.

#### Methods

| Method | Description |
|--------|-------------|
| `clip_weights()` | Clamps the LUT entries to `[-1, 1]`. Only active in training mode and if `clamp_luts=True`. Call after every optimizer step. |
| `freeze_mapping()` | Replaces a (subset) learnable mapping by the fixed `argmax` selection, which is stored as an `int32` tensor of shape `(output_size, n)`. No-op for fixed mappings.|
| `export_binary_luts()` | Returns the LUT contents as booleans (`luts > 0`), the 1-bit tables used at inference. |

### `freeze`

```python
dwn.freeze(model=model)
```

Calls `freeze_mapping()` on every `LUTLayer` in `model` and returns the model. After freezing, the layers behave like fixed-mapping layers: the mapping scores (and, for the subset mapping, the pools) are removed, so the model is smaller. A frozen layer is no longer learnable.

### `LearnableMapping`

```python
LearnableMapping(input_size, output_size, tau=0.001)
```

| Argument | Description |
|----------|-------------|
| `input_size` | Input size. |
| `output_size` | Output size, typically n * N (LUT rank * number of LUTs in the next layer). |
| `tau` | Softmax temperature for the backpropagation, i.e., `lm_tau` |

### `SubsetLearnableMapping`

```python
SubsetLearnableMapping(input_size, output_size, n, pool_size=100, tau=0.001, mode='random')
```

| Argument | Description |
|----------|-------------|
| `input_size` | Input size. |
| `output_size` | Number of LUTs. |
| `n` | Inputs per LUT (rank). |
| `pool_size` | Candidate bits per LUT (`<= input_size`). |
| `tau` | Softmax temperature for the backpropagation, i.e., `lm_tau`, (softmax over each pool). |
| `mode` | Pool construction: `'random'`, `'block'` or `'strided'`. |

- `weights`: learnable scores of shape `(output_size, pool_size, n)`.
- `pool_idx`: buffer of shape `(output_size, pool_size)` with the input indices available to each LUT.

### `GroupSum`
Applies a grouped sum (popcount) over the outputs, followed by softmax temperature:

```python
GroupSum(k, tau=1, randperm=False)
```

| Argument | Description |
|----------|-------------|
| `k` | Number of groups (e.g., classes). |
| `tau` | Temperature, the group sums are divided by it. |
| `randperm` | If an int is given, the inputs are shuffled by a fixed random permutation of that size. |

Inputs are zero-padded if the output size is not divisible by `k`.

---
