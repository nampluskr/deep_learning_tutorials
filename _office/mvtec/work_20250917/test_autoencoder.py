```python
import torch
from test_autoencoder import get_config, get_dataloaders, AutoEncoder
from test_autoencoder import model_test
```

```python
config = get_config(category="tile")
config.batch_size = 4
train_loader, test_loader = get_dataloaders(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(latent_dim=config.latent_dim).to(device)

print(config.weight_path)
state = torch.load(config.weight_path, map_location=device)
model.load_state_dict(state['model_state_dict'], strict=False)
model_test(model, test_loader, device)
```

```python
config = get_config(category="grid")
config.batch_size = 4
train_loader, test_loader = get_dataloaders(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(latent_dim=config.latent_dim).to(device)

print(config.weight_path)
state = torch.load(config.weight_path, map_location=device)
model.load_state_dict(state['model_state_dict'], strict=False)
model_test(model, test_loader, device)
```
