# Slot-DWN

👋 Welcome! This is the official implementation of the manuscript **"Object-Centric Differentiable Logic Networks for Abstract Visual Reasoning"**.

**Slot-DWN** is a hybrid architecture combining a perception backbone, i.e., a pretrained Slot Attention encoder which splits each image into object-centric *slots*, with a Multi-Head Differentiable Weightless Network (MH-DWN), which reasons over these object slots using only lookup tables (LUTs), instead of dense weights.

**Abstract Visual Reasoning (AVR) Tasks**

- 🧩 **Abstract Reasoning Tasks (ART):** Simple visual reasoning tasks, measuring extreme OOD generalization, on heldout shapes.
- 🔍 **Odd-Orientation-Out (O3):** Finding the *odd* panel whose object has a different orientation than the others, in an easy and a hard variant.
- 🧮 **Relational Matrices (RM):** RPM-like problems, where multiple relationships among the objects in a 3x3 grid must be inferred. In addition, similar to ART tasks, RM tasks also measure OOD generalization on heldout shapes.

This code is tested with Python $\geq$ 3.11. We used Pytorch 2.13.0 compiled with CUDA version 13.0 for all the experiments. You can install the required packages via pip:

```
pip install -r requirements.txt
```

## Repository Structure

```
.
├── ART/              # Abstract Reasoning Tasks (same_diff, RMTS, dist3, identity_rules)
├── O3/               # Odd-Orientation-Out tasks (O3-Easy, O3-Hard)
├── RM/               # Relational Matrices task + slot attention pretraining for RM
├── configs/          # YAML configs per task
├── dwn_source/       # Updated Differentiable Weightless Network (torch_dwn) source code, with sparse learnable mapping
├── mh_dwn/dwn.py     # Multi-Head implementation of DWNs
├── main.py           # Main script to run tasks
└── util.py
```

## Installation (for torch_dwn package)

Locally build and install the `torch_dwn` package from `dwn_source`:

```bash
cd dwn_source
pip install . --no-build-isolation
```

See `dwn_source/torch_dwn/build_dwn.sh` for an example SLURM build script and `dwn_source/torch_dwn/README.md` for detailed usage and example.

---

## Tasks

- [Abstract Reasoning Tasks (ART)](#1-abstract-reasoning-tasks-art)
- [Odd-Orientation-Out (O3)](#2-odd-orientation-out-o3)
- [Relational Matrices (RM)](#3-relational-matrices-rm)

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

The shape images must be downloaded from the [OCRA repository](https://github.com/Shanka123/OCRA/tree/main/imgs) and must be placed under `ART/data/`:

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
| `m_holdout` | Number of shapes withheld during training, default 95 |
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

Config options for O3 tasks (the slot attention, optimization and MH-DWN options are the same as for ART-SD task). Extra parameters needed for O3:

| Key | Description |
|-----|-------------|
| `path` | Directory with the `.npz` data files |
| `data_name` | `orientation` or `orientation-2obj`, files are named `<data_name>-o3-{train,test}.npz` |
| `split_seed` | Optinal seed for fixing the train/validation split. Otherwise it is random |
| `patience` / `min_delta` | Early stopping on the validation loss |
| `train_size`| Number of training samples, default 1000|
| `val_size`|  Number of validation samples, used for early stopping, default 200|

> With a single object (`o3_easy`) there are no cross-object pairs, so the slot is paired with itself as input to the `U_rel` layer.

---

## 3. Relational Matrices (RM)

Each sample is a 3x3 grid of shapes whose attributes (shape, rotation, zoom, grayscale) follow row, column or diagonal rules, with the bottom-right cell missing. Five full candidate grids are given (1 correct, 4 distractors), and the model must pick the right one. The number of relations that are changed in a problem is `num_relations` (1, 2 or 3).

The scripts are under `RM/`. Every candidate grid is encoded by a pretrained Slot Attention autoencoder (`RM/slot_ae.py`, pretrained checkpoint `RM/sae_checkpoints/enc64-dec8-k11-aug-final.pth.tar`) and scored independently by the MH-DWN model (`mh_dwn/dwn.py`). The candidate with the highest score is the prediction.

Two sets of problems are used:

- **G-set** (`--split g-set`): shapes seen in training. It is split into a train and a test part.
- **S-set** (`--split s-set`): problems built from held-out shapes only, to measure generalization to unseen shapes. It is used as the transfer set after training.

### Data

The data generators are under `RM/datagen/`. `generate.sh` is an example SLURM script, which generates all three sets for 1, 2 and 3 relations. Example generation:

```bash
cd RM/datagen

# G-set train and test samples
python generate_grid_choice.py --root $DATA/RM_problems --num-samples 5000 --num-relations 1 --split g-set --ext train
python generate_grid_choice.py --root $DATA/RM_problems  --num-samples 500 --num-relations 1 --split g-set --ext test

# S-set samples
python generate_grid_choice.py --root $DATA/RM_problems  --num-samples 500 --num-relations 1 --split s-set --ext test
```

`--split` selects the shapes (`g-set` uses the train shapes, `s-set` the held-out ones), and `--ext` (`train` or `test`) is dataset. The samples are saved as `<root>/<num_relations>/<split>/<ext>/<i>.npz`, e.g., `RM_problems/1/g-set/train/0.npz`. Repeat each command with `--num-relations 2` and `3` for the other problems.

Change `--root` to the folder you want. Set the path also in `configs/RM.yaml`, respectively.

### Training

Same as the other tasks, it is run from the repository root. The number of relations is set by `num_relations` in `configs/RM.yaml`:

```bash
python main.py --task RM
```

| `--task` | Config               | Training script        |
|----------|----------------------|------------------------|
| `RM`     | `configs/RM.yaml`    | `RM/train_rm_dwn.py`   |

The test accuracy for G-set is reported every `eval_interval` epochs, and the accuracy on the S-set at the end.

Config options for RM (the MH-DWN options are listed below):

| Key | Description |
|-----|-------------|
| `num_relations` | Number of relations in the problems: 1, 2 or 3 |
| `slot_ae.ckpt` | Pretrained slot autoencoder checkpoint. Its model config is read from the checkpoint |
| `num_obj` | Actual number of objects, passed to the DWN, i.e., K'. |
| `seed` | Seed of the (frozen) slot attention encoder, used only for its deterministic initial slots |
| `data.path` | Root folders containing G-set train & test and S-set. |
| `data.num_workers` | Data loader workers |
| `dwn` | MH-DWN hyperparams, see below |
| `optim` | `batch_size`, `learning_rate`, `min_lr` (cosine annealing) and `num_epochs` |
| `train` | `device`, `no_cuda`, `log_interval` and `eval_interval` (epochs between test evaluations) |

### Training the slot autoencoder from scratch

This step is optional, if you want to pretrain the slot autoencoder from scratch with `RM/train_rm_sae.py`:

1. Generate images with randomly placed shapes. They are written to `<root>/<split>/<ext>/`, e.g., `$DATA/deepiq_random/g-set/train/0.npz`:

   ```bash
   cd RM/datagen
   python generate_random.py --root $DATA/deepiq_random --num-samples 150000 --split g-set --ext train
   python generate_random.py --root $DATA/deepiq_random --num-samples 5000 --split g-set --ext test
   ```

2. Set `data.path` in `RM/rm_slot_ae.yaml` to the `--root` above.

3. Train, from `RM/`:

   ```bash
   cd RM
   python train_rm_sae.py
   ```

   | Argument | Description |
   |----------|-------------|
   | `--config` | Path of the yaml config, default `rm_slot_ae.yaml` |
   | `--ckpt` | Checkpoint to load the model (its saved config and weights) from. Training resumes from its saved iteration |
   | `--visualize` | Skip training: load `--ckpt` and save slot reconstruction figures |
   | `--vis-out` / `--vis-n` | Output prefix of the figures, and number of random test samples |

   The checkpoints `<model_name>-best.pth.tar` and `<model_name>-final.pth.tar` are saved to `ckpt_dir`, and the reconstructions to `log_dir/vis`.

4. To use the pretrained autoencoder you trained, set `slot_ae.ckpt` in `configs/RM.yaml` to its checkpoint.

---
MH-DWN options:

| Key | Description |
|-----|-------------|
| `dwn_heads` | Number of LUT heads per layer |
| `dwn_hid` | LUTs per layer, (after `U_rel` layer), halved in each layer after|
| `dwn_num_layers` | Number of DWN layers |
| `dwn_levels` | Thermometer encoding levels |
| `tau` | Pre-softmax temperature of the output (GroupSum) layer |
| `n` | LUT input arity |
| `lm_tau` | Softmax temperature for learnable mappings |
| `first_layer_heads` | Number of LUT heads in the first (`U_rel`) layer |
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
