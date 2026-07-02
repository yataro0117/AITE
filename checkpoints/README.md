# Model Checkpoints

This directory holds model weights. Two sources populate it:

* **Released pretrained weights** — trained checkpoints from the paper, published
  via **[GitHub Releases](https://github.com/yataro0117/AITE/releases)**
  (TODO: upload after paper publication). Not included in the repository.
* **Locally trained checkpoints** — `scripts/train.py` saves here by default
  (`--save_dir checkpoints/`).

Weight files (`*.pt`) are git-ignored; only this README and `.gitkeep` are tracked.

## Expected layout

After downloading and unzipping the release archive:

```
checkpoints/
├── unipen/
│   ├── cnn3_{1a,1b,1c}.pt
│   ├── cnn4_{1a,1b,1c}.pt
│   ├── blstm2_{1a,1b,1c}.pt
│   └── transformer_{1a,1b,1c}.pt
└── casia/
    ├── cnn1d/best.pt
    ├── blstm2/best.pt
    └── transformer/best.pt
```

Each `.pt` file is a PyTorch state dict (`torch.save(model.state_dict(), path)`).
Load with:

```python
import torch
from aite.models.unipen import get_unipen_model

model = get_unipen_model("cnn3", dataset="1a")
model.load_state_dict(torch.load("checkpoints/unipen/cnn3_1a.pt", map_location="cpu"))
model.eval()
```

## Training configurations

| Model | Dataset | Optimizer | LR | Epochs | Seed |
|-------|---------|-----------|-----|--------|------|
| CNN-3 | Unipen 1A/1B/1C | Adam | 1e-3 | — | 42 |
| CNN-4 | Unipen 1A/1B/1C | Adam | 1e-3 | — | 42 |
| BLSTM-2 | Unipen 1A/1B/1C | Adam | 1e-3 | — | 42 |
| Transformer | Unipen 1A/1B/1C | Adam | 1e-3 | — | 42 |
| CNN-1D | CASIA-OLHWDB1.1 | AdamW | 1e-3 | 80 | 42 |
| BLSTM | CASIA-OLHWDB1.1 | AdamW | 1e-3 | 80 | 42 |
| Transformer | CASIA-OLHWDB1.1 | AdamW | 1e-3 | 100 | 42 |

See `configs/` for the full hyperparameter files used for attack experiments.
