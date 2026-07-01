# Hierarchical Implicit Neural Emulators

Welcome to our Futurama: (Future-dependent) hierarchical implicit emulators! 

This is the official implementation for our NeurIPS 2025 paper: Hierarchical Implicit Neural Emulators (Operators) ([arxiv version](https://arxiv.org/abs/2506.04528)). You can use it directly for modeling 2D PDE dynamics, as we built the architecture based on UNet2d and Fourier convolutions. But the main method is general enough to any autoregressive modeling.

### Command for running experiments
For training the model using 4 GPUs.
```
bash experiments/srun.sh
```

To test the model using a single GPU.
```
bash experiments/eval.sh
```
For customization, you need to rewrite the dataloader.py file to load your own data. For different dynamics, you need to check the setup in the model.py, to tune the hyperparameters like the blocks number, the Fourier modes, and the channel width of the network.
