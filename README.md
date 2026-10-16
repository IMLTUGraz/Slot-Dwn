# <Project Name>

<!-- TODO: project description / abstract here. -->

## Repository Structure

```
.
├── ART/              # Abstract Reasoning Tasks (same_diff, RMTS, dist3, identity_rules)
├── configs/          # YAML configs per task
├── dwn_source/       # Differentiable Weightless Network (torch_dwn) source integrated with proposed sparse learnable mapping
├── main.py           # Main script to run tasks
├── MH_DWN.py         # Slot-based multi-head DWN model
└── util.py
```

## Installation

Locally build and install the `torch_dwn` package under `dwn_source`:

```bash
cd dwn_source/torch_dwn
pip install . --no-build-isolation
```

See `dwn_source/torch_dwn/build_dwn.sh` for an example SLURM build script.

---

## Tasks

- [Abstract Reasoning Tasks (ART)](#1-abstract-reasoning-tasks-art)

---

## 1. Abstract Reasoning Tasks (ART)

The ART code lives in `ART/`. It contains four tasks, each with its own training script:

| Task             | Script                  | Data generator            |
|------------------|-------------------------|---------------------------|
| Same-Different   | `ART/train_sd.py`       | `ART/data/same_diff.py`   |
| RMTS             | `ART/train_rmts.py`     | `ART/data/RMTS.py`        |
| Distribution-of-3 | `ART/train_dist3.py`   | `ART/data/dist3.py`       |
| Identity Rules   | `ART/train_identity.py` | `ART/data/identity_rules.py` |

The scripts use a pretrained Slot Attention autoencoder (`ART/slot_ae_ocra.py`, checkpoint `ART/ocra-art.pth.tar`) as the object-centric encoder and a multi-head slot DWN (`MH_DWN.py`) as the reasoner.

### Data

The shape images must be downloaded from the [OCRA repository](https://github.com/Shanka123/OCRA/imgs/) and must placed under `ART/data/`:

```bash
git clone https://github.com/Shanka123/OCRA.git /tmp/OCRA
mkdir -p ART/data
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

Training is launched through `main.py` from the repository root. Pick a task with `--task`, its hyperparameters are read from the matching file in `configs/`:

```bash
# Same-Different
python main.py --task same_diff

# Relational Match-to-Sample (RMTS)
python main.py --task rmts

# Distribution of 3
python main.py --task dist3

# Identity Rules
python main.py --task identity_rules
```

| `--task`         | Config                        | Training script         |
|------------------|-------------------------------|-------------------------|
| `same_diff`      | `configs/same_diff.yaml`      | `ART/train_sd.py`       |
| `rmts`           | `configs/rmts.yaml`           | `ART/train_rmts.py`     |
| `dist3`          | `configs/dist3.yaml`          | `ART/train_dist3.py`    |
| `identity_rules` | `configs/identity_rules.yaml` | `ART/train_identity.py` |

To use a different config you can change the respective yaml files.


Config options:

| Key | Default | Description |
|-----|---------|-------------|
| `train_gen_method` / `test_gen_method` | `full_space` (`subsample` for the test split of `(m=95)` heldout regime of `rmts` and `identity_rules`) | `full_space` or `subsample` |
| `n_shapes` | 100 | Total number of shapes available |
| `m_holdout` | 95 | Number of shapes withheld during training |
| `num_slots` | 6 | Number of slots in Slot Attention |
| `batch_size` | 16 | Batch size |
| `num_epochs` | 100 | Training epochs |
| `learning_rate` | 1e-2 | Learning rate |

MH-DWN options:

| Key | Default | Description |
|-----|---------|-------------|
| `dwn_heads` | 2 | Number of LUT heads per layer |
| `dwn_hid` | 256 (`same_diff`), 512 (`rmts`), 1024 (`dist3`, `identity_rules`) | Hidden LUTs per layer |
| `dwn_num_layers` | 2 | Number of DWN layers |
| `dwn_levels` | 5 | Thermometer encoding levels |
| `n` | 4 | LUT input arity |
| `mapping_type` | `paired` | `one-to-one` or `paired` |
| `use_subset_lm` | `true` | Use the sparse (subset) learnable mapping |
| `subset_lm_mode` | `strided` | `random`, `strided` or `block` |

See the `configs/*.yaml` for the full list.

---

## Citation

<!-- TODO -->

## Acknowledgements

- ART data and the Slot Attention checkpoint are from [OCRA](https://github.com/Shanka123/OCRA).
- The `torch_dwn` library is included under `dwn_source/`, which is built upon the original DWN implementation from (https://github.com/alanbacellar/DWN).
