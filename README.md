# <Project Name>

<!-- TODO: project description / abstract here. -->

## Repository Structure

```
.
├── ART/              # Abstract Reasoning Tasks (same_diff, RMTS, dist3, identity_rules)
├── O3/               # Odd-Orientation-Out tasks (O3-Easy, O3-Hard)
├── configs/          # YAML configs per task
├── dwn_source/       # Updated Differentiable Weightless Network (torch_dwn) source code, with sparse learnable mapping
├── mh_dwn/dwn.py     # Multi-Head implementation of DWNs
├── main.py           # Main script to run tasks
└── util.py
```

## Installation

Locally build and install the `torch_dwn` package under `dwn_source`:

```bash
cd dwn_source/torch_dwn
pip install . --no-build-isolation
```

See `dwn_source/torch_dwn/build_dwn.sh` for an example SLURM build script and `dwn_source/README.md` for detailed usage and example.

---

## Tasks

- [Abstract Reasoning Tasks (ART)](#1-abstract-reasoning-tasks-art)
- [Odd-Orientation-Out (O3)](#2-odd-orientation-out-o3)

---

## 1. Abstract Reasoning Tasks (ART)

The ART task scripts are under `ART/` directory. It contains four tasks, each with its own training script:

| Task             | Script                  | Data generator            |
|------------------|-------------------------|---------------------------|
| Same-Different   | `ART/train_sd.py`       | `ART/data/same_diff.py`   |
| RMTS             | `ART/train_rmts.py`     | `ART/data/RMTS.py`        |
| Distribution-of-3 | `ART/train_dist3.py`   | `ART/data/dist3.py`       |
| Identity Rules   | `ART/train_identity.py` | `ART/data/identity_rules.py` |

The scripts use a pretrained Slot Attention autoencoder (`ART/slot_ae_ocra.py`, checkpoint `ART/ocra-art.pth.tar`) as the object-centric encoder and a multi-head slot DWN (`mh_dwn/dwn.py`) as the reasoner.

### Data

The shape images must be downloaded from the [OCRA repository](https://github.com/Shanka123/OCRA/tree/main/imgs) and must placed under `ART/data/`:

```bash
git clone https://github.com/Shanka123/OCRA.git /tmp/OCRA
cp -r /tmp/OCRA/imgs ART/data/imgs
```

The final layout should be:

```
ART/data/imgs/
├── 0.png
├── cropped0.npy
├── resizedcropped0.npy
└── ...
```

### Training

Training is launched through `main.py` from the repository root. Pick a task with `--task`, its hyperparameters are read from the matching file in `configs/`, e.g.,:

```bash
# Same-Different
python main.py --task same_diff

# Relational Match-to-Sample (RMTS)
python main.py --task rmts
```

| `--task`         | Config                        | Training script         |
|------------------|-------------------------------|-------------------------|
| `same_diff`      | `configs/same_diff.yaml`      | `ART/train_sd.py`       |
| `rmts`           | `configs/rmts.yaml`           | `ART/train_rmts.py`     |
| `dist3`          | `configs/dist3.yaml`          | `ART/train_dist3.py`    |
| `identity_rules` | `configs/identity_rules.yaml` | `ART/train_identity.py` |

To use a different config you can change the respective yaml files.


Config options for ART tasks:

| Key | Description |
|-----|-------------|
| `train_gen_method` / `test_gen_method` | Sample generating approach: `full_space` or `subsample` |
| `n_shapes` | Total number of shapes available, default 100|
| `m_holdout` | Number of shapes withheld during training |
| `num_slots` | Number of slots in pretrained Slot Attention|
| `batch_size` | Batch size |
| `num_epochs` | Training epochs |
| `learning_rate` | Learning rate |

See the `configs/*.yaml` for the full list.

---

## 2. Odd-Orientation-Out (O3)

In each sample, three panels (synthetic dSprites-like shapes) are shown, and the model must pick the odd panel whose object has a different orientation. The scripts are under `O3/`:

| Task      | Dataset name            | Config                  | Description |
|-----------|-------------------------|-------------------------|-------------|
| O3-easy   | `orientation`           | `configs/o3_easy.yaml`  | One object per panel |
| O3-hard   | `orientation-2obj`      | `configs/o3_hard.yaml`  | Two objects per panel, only one of them is rotated in the odd panel |

Both tasks use `O3/train_o3.py`, the same pretrained Slot Attention encoder as ART (`ART/ocra-art.pth.tar`), and the multi-head slot DWN (`mh_dwn/dwn.py`). 
Every panel is scored independently and the panel with the highest score is the prediction.

### Data

The data is already available under `O3/data/` as .npz files:

```
O3/data/
├── orientation-o3-train.npz
├── orientation-o3-test.npz
├── orientation-2obj-o3-train.npz
└── orientation-2obj-o3-test.npz
```

To generate it again (or with different sizes/shapes), run `O3/dsprites.py`. It writes the four files above into the current directory:

```bash
cd O3/
python dsprites.py
```

### Training

Same as ART, from the repository root:

```bash
# O3-easy
python main.py --task o3_easy

# O3-hard
python main.py --task o3_hard
```

| `--task`  | Config                 | Training script    |
|-----------|------------------------|--------------------|
| `o3_easy` | `configs/o3_easy.yaml` | `O3/train_o3.py`   |
| `o3_hard` | `configs/o3_hard.yaml` | `O3/train_o3.py`   |

Config options for O3 tasks (the slot attention, optimization and MH-DWN options are the same as for ART-SD task).

| Key | Description |
|-----|-------------|
| `path` | Directory with the `.npz` data files |
| `data_name` | `orientation` or `orientation-2obj`, files are named `<data_name>-o3-{train,test}.npz` |
| `split_seed` | Seed for the train/validation split, random if not given |
| `patience` / `min_delta` | Early stopping on the validation loss |

> With a single object (`o3_easy`) there are no cross-object pairs, so the slot is paired with itself as input to the `U_rel` layer.

---
MH-DWN options:

| Key | Description |
|-----|-------------|
| `dwn_heads` | Number of LUT heads per layer |
| `dwn_hid` | LUTs per layer, (after `U_rel` layer), halved in each layer after|
| `dwn_num_layers` | Number of DWN layers |
| `dwn_levels` | Thermometer encoding levels |
| `n` | LUT input arity |
| `mapping_type` | `one-to-one` or `paired` mapping for binary slot embeddings |
| `use_subset_lm` | Use the sparse (subset) learnable mapping (True/False)|
| `subset_lm_mode` | `random`, `strided` or `block` |
| `pool_size` | The pool size effective when `use_subset_lm` is true. It can be explicitly given, otherwise it will be calculated based on the input size |
---

## Citation

<!-- TODO -->

## Acknowledgements

- ART data and the pretrained slot attention checkpoint (which is also used for O3 tasks) are from [OCRA](https://github.com/Shanka123/OCRA).
- The `torch_dwn` library is included under `dwn_source/`, which is built upon the original DWN implementation from (https://github.com/alanbacellar/DWN).
