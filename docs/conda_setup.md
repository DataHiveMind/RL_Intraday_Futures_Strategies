# Conda environment: create, initialize, activate, and troubleshooting

This document describes how to initialize Conda for your shell, create the project's conda environment from `environment.yml`, activate it, update it, and common troubleshooting steps.

## Quick one-time shell initialization

If `conda activate` fails with `CondaError: Run 'conda init' before 'conda activate'`, run the one-off hook to initialize Conda in the current shell session (no restart required):

```bash
eval "$(conda shell.bash hook)"
conda activate rl-intraday-futures
```

This sets up the shell integration for the current session only.

## Persistent setup (recommended)

To configure your shell permanently so future terminals are ready to use Conda, run:

```bash
conda init bash
# then reload your shell config
source ~/.bashrc
# now you can activate
conda activate rl-intraday-futures
```

If your Conda installation is in a non-standard prefix, you can source the Conda profile script directly:

```bash
. /opt/conda/etc/profile.d/conda.sh   # <--- adjust path to your conda install
conda activate rl-intraday-futures
```

## Create the environment from `environment.yml`

From the repository root run:

```bash
conda env create -f environment.yml
conda activate rl-intraday-futures
```

If the environment name in `environment.yml` conflicts with an existing env, create with a different name:

```bash
conda env create -f environment.yml -n rl-intraday-futures-local
conda activate rl-intraday-futures-local
```

## Update an existing env from the YAML

To update the environment after changes to `environment.yml`:

```bash
conda env update -f environment.yml -n rl-intraday-futures --prune
```

`--prune` removes packages not listed anymore.

## GPU / PyTorch notes

- The `environment.yml` in this repo installs `torch` via `pip` for simplicity. For GPU-enabled PyTorch, use the official PyTorch install instructions to get the correct CUDA build. Example (CUDA 11.8):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

- Alternatively, use `mamba` (faster) to install GPU packages:

```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Verify installation

Run quick checks to ensure key packages are available:

````bash
python -c "import sys, pkgutil; print('python', sys.version)
import numpy as np; print('numpy', np.__version__)
import pandas as pd; print('pandas', pd.__version__)
import torch; print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
"

or separately:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
conda list | grep -E 'numpy|pandas|torch|stable-baselines3|gymnasium'
````

If you expect GPU but `torch.cuda.is_available()` is `False`, verify `nvidia-smi` is present and drivers are installed on the host (containers require NVIDIA runtime).

## VS Code integration

- Open the Command Palette -> `Python: Select Interpreter` -> choose the Conda environment `rl-intraday-futures`.
- Restart any running terminals in VS Code after `conda init` or after creating the env so the new shell integration is recognized.

## Troubleshooting

- `conda: command not found`: ensure Conda is installed and either added to your PATH or source `conda.sh` as shown above.
- `CondaActivationError` or `CommandNotFoundError`: restart the terminal or run `eval "$(conda shell.bash hook)"` in the current session.
- Package install failures: try `mamba` or install heavy binary packages (PyTorch, XGBoost) via conda channels to avoid wheel compilation.

## Removing the environment

```bash
conda env remove -n rl-intraday-futures
```

## Fast alternative: use `mamba`

`mamba` is a drop-in replacement for `conda` that is substantially faster for resolving packages. Install it with:

```bash
conda install mamba -n base -c conda-forge
```

Then run create/update commands with `mamba`:

```bash
mamba env create -f environment.yml
mamba env update -f environment.yml --prune
```

---
