## TOTO

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from dataloader import get_dataloaders
from main import get_device
from utils import show_model_info, get_thresholds, evaluate_thresholds, model_test
from trainer import AutoEncoderTrainer

from model_stfpm import STFPMModel

device = get_device()
model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18").to(device)
model_type = "stfpm-resnet18"
```

```python
category = "tile"   # ["carpet", "grid", "leather", "tile", "wood"]
_, test_loader = get_dataloaders(root="/home/namu/myspace/NAMU/datasets/mvtec", 
    category=category, batch_size=4, img_size=256)
weight_path = os.path.join("./experiments", category, f"{category}_{model_type}_epochs-50.pth") 
model.load_state_dict(torch.load(weight_path, map_location=device))

model_test(model, test_loader, device)
```
