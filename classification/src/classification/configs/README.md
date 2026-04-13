# Configs

Each config file is a dataclass that inherits from `BaseConfig`.

## Naming convention for saved models

```
{author}_{config_name}_{date}_{val_acc}_{test_acc}/
```

Example: `mateo_resnet32_20260413_0982_0874/`

## Available configs

| Config | N_MEL | Notes |
|---|---|---|
| `BaseConfig` (base) | 64 | Default, matches current best results |
| `ResNet32Config` (resnet32) | 32 | Smaller mel resolution, faster training |

## Adding a new config

```python
# classification/src/classification/configs/myconfig.py
from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class MyConfig(BaseConfig):
    N_MEL: int = 48
    DROPOUT_RATE: float = 0.25
    ACTIVATION: str = "swish"
```

Then register it in `training/train.py`:
```python
CONFIG_REGISTRY["myconfig"] = MyConfig
```

## Key design decisions

- `n_frames` is computed EXCLUSIVELY from Python DSP params via `compute_n_frames()`.
  The MCU firmware `config.h` must NOT be modified for this.
- `BACKGROUND_DATA_DIR`: set this to activate background class (5th class).
  Leave empty to use 4-class mode (chainsaw/fire/fireworks/gunshot).
- `ACTIVATION`: one of `relu | gelu | swish | mish | leaky_relu`
