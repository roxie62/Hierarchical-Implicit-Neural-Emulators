# Hierarchical Implicit Neural Emulators

Welcome to our Futurama: (Future-dependent) hierarchical implicit emulators! 

This is the official implementation for our NeurIPS 2025 paper: Hierarchical Implicit Neural Emulators (Operators) ([arxiv version](https://arxiv.org/abs/2506.04528)). You can use it directly for modeling 2D PDE dynamics, as we built the architecture based on UNet2d and Fourier convolutions. But the main method is general enough to any autoregressive modeling.

TLDR: Neural PDE solvers are powerful but tend to accumulate error over long rollouts, drifting away from stable, physically consistent trajectories. We introduce a **multiscale implicit neural emulator** that improves long-term accuracy by conditioning each prediction on a hierarchy of lower-dimensional representations of *future* states. Drawing on the stability of implicit time-stepping schemes in classical numerical methods, the model looks several steps ahead at increasing compression rates and uses those coarse forecasts to refine the next-timestep prediction. By actively adjusting the temporal downsampling ratios, it captures dynamics across multiple granularities and enforces long-range temporal coherence.

## How it works

The emulator predicts the next high-resolution state autoregressively, but each step is conditioned on a hierarchy of coarser (spatially downsampled) summaries of where the system is heading. This mirrors implicit time-stepping, which solves for a future state that satisfies a consistency condition rather than extrapolating purely from the past.

Training jointly supervises three coupled prediction cases, sampled on a randomized schedule (controlled by `--ratio`):

1. Predict the **mid-resolution** future summary `z¹` from the current state.
2. Predict the **low-resolution** future summary `z²` given the current state and `z¹`.
3. Predict the **next full-resolution** state given the current state, `z¹`, and `z²`.

A single Fourier-augmented U-Net (`ufno.py`) produces all three resolutions in one forward pass, injecting the coarse predictions back into the encoder. An exponential-moving-average (EMA) copy of the weights is used at inference, and long rollouts are produced by repeatedly applying the sampler.

## Repository structure

```
├── main.py                 # Entry point: argument parsing, training/eval loop, checkpointing
├── model.py                # FutureINN: the hierarchical multiscale emulator (training + sampling)
├── ufno.py                 # Fourier-augmented U-Net backbone (spectral convolutions)
├── dataloader.py           # DistributedPreloadDataset: consecutive-frame sequence loader
├── engine_mar.py           # train_one_epoch, autoregressive rollout, MSE evaluation, plotting
├── experiments/
│   ├── srun.sh             # 4-GPU training launch (SLURM)
│   └── eval.sh             # single-GPU evaluation launch
└── util/
    ├── py2d.py             # 2D Navier–Stokes dataset + spectral vorticity/streamfunction/velocity ops
    ├── fourier.py          # Spectral (Fourier) convolution layers
    ├── data_losses.py      # Loss functions
    ├── lr_sched.py         # Learning-rate schedule
    └── misc.py             # Distributed-training, checkpoint, and logging helpers
```

## Installation

```bash
git clone https://github.com/roxie62/Hierarchical-Implicit-Neural-Emulators.git
cd Hierarchical-Implicit-Neural-Emulators

# create an environment (conda or venv)
conda create -n hine python=3.10 -y
conda activate hine

# install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch torchvision

# remaining dependencies
pip install numpy scipy einops matplotlib opencv-python pandas tqdm torch_fidelity
```

The code uses PyTorch Distributed (`torchrun`) and mixed-precision autocast; a CUDA-capable GPU is required for training.

## Data preparation

The dataset loader (`util/py2d.py: NSdataset`) reads snapshots of **2D forced homogeneous isotropic turbulence** — i.e., trajectories of the 2D Navier–Stokes equations in vorticity form on a `2π × 2π` periodic domain.

### File format

Each timestep is a MATLAB `.mat` file containing the 2D vorticity field under the key `Omega` (shape `nx × ny`):

```
train/
├── 0.mat        # {'Omega': ndarray of shape (nx, ny)}
├── 5.mat
├── 10.mat
└── ...
```

- Files are **named by integer timestep index**. The loader keeps every 5th index (`time_step = 5`) and asserts the kept indices are evenly spaced.
- Fields are bilinearly resampled to `--img_size` (default `256`) and divided by a fixed normalization constant (`10`) at load time.
- Consecutive files must correspond to consecutive timesteps so that stacked frames form a valid trajectory.

### Directory layout

`NSdataset` looks for `train/`, `validation/`, and `eval/` **one level above the directory you launch from** (they are treated as siblings of your working directory). For example, if you run from `.../Hierarchical-Implicit-Neural-Emulators`, place the data at:

```
parent/
├── Hierarchical-Implicit-Neural-Emulators/   # <- you launch from here
├── train/
├── validation/
└── eval/
```

Symlinks work well if your data lives elsewhere.

### Generating the data

Bring your own pseudo-spectral 2D Navier–Stokes solver to produce the trajectories. The spectral utilities in `util/py2d.py` (`Omega2Psi`, `Psi2UV`, `initialize_wavenumbers_rfft2`) follow the standard vorticity–streamfunction–velocity formulation, so any solver in that convention is compatible. For each simulation, export the vorticity field `Omega` at each timestep to `<index>.mat`, then split trajectories across the `train/`, `validation/`, and `eval/` folders.

## Training

**On a SLURM cluster (4 GPUs):**

```bash
bash experiments/srun.sh
```

**Directly on a single multi-GPU node** (the SLURM script assumes `scontrol`/`SLURM_*` are available, so use this outside a cluster):

```bash
torchrun --nnodes=1 --nproc_per_node=4 main.py \
  --img_size 256 --batch_size 32 \
  --num_of_frame 4 --ds_factor 8 --ds_factor_low 32 \
  --output_dir output --resume output
```

Checkpoints are written to `--output_dir` as `checkpoint-last.pth` and periodic `checkpoint-<epoch>.pth`; training resumes automatically if a checkpoint is found there.

## Evaluation

**On a single GPU:**

```bash
bash experiments/eval.sh
```

**Or directly**, loading a specific checkpoint (e.g. epoch 700):

```bash
torchrun --nnodes=1 --nproc_per_node=1 main.py \
  --img_size 256 --batch_size 32 \
  --num_of_frame 4 --ds_factor 8 --ds_factor_low 32 \
  --output_dir output --resume output \
  --evaluate --resume_epoch 700
```

Evaluation rolls the model out autoregressively over the test set and reports mean-squared error as a function of forecast horizon.

## Key hyperparameters

| Argument | Reference value | Meaning |
| --- | --- | --- |
| `--num_of_frame` | `4` | Consecutive frames per training sample. **Must be 4** for the default 3-level hierarchy. |
| `--ds_factor` | `8` | Spatial downsampling factor for the mid-resolution future summary `z¹` (→ `img_size / ds_factor`). |
| `--ds_factor_low` | `32` | Spatial downsampling factor for the low-resolution future summary `z²` (→ `img_size / ds_factor_low`). |
| `--ratio` | `0.85` | Fraction of samples trained on the full autoregressive case vs. the two warm-up cases. |

Outputs (plots, per-horizon MSE CSVs, checkpoints) are written under `--output_dir`.

## Adapting to your own dynamics

The method is not specific to fluid flow. To model a different system:

- **Data:** rewrite `dataloader.py` (and `util/py2d.py` if you keep the `NSdataset` interface) to yield tensors of shape `(num_of_frame, H, W)` from your own source.
- **Architecture:** in `model.py` / `ufno.py`, tune the backbone to your dynamics — the number of U-Net blocks, the retained Fourier modes, and the channel width.
- **Hierarchy:** adjust `--ds_factor`, `--ds_factor_low`, and `--num_of_frame` to set how far ahead and at what compression the future summaries are formed.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{NEURIPS2025_6aa49679,
 author = {Jiang, Ruoxi and Zhang, Xiao and Jakhar, Karan and Lu, Peter Y. and Hassanzadeh, Pedram and Maire, Michael and Willett, Rebecca},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Hierarchical Implicit Neural Emulators},
 volume = {38},
 year = {2025}
}

```
