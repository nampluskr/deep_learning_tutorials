```python
import torch
from dataloader import get_dataloaders
from model_autoencoder import AutoEncoder
from trainer import AutoEncoderTrainer
from main import get_config, show_model_info

category = "tile"
num_epochs = 50
output_dir = f"./results/{category}"
latent_dim = 256

config = get_config("autoencoder", category, num_epochs, output_dir, latent_dim=latent_dim)
train_loader, test_loader = get_dataloaders(config)
model = AutoEncoder(latent_dim=latent_dim)
show_model_info(model)

trainer = AutoEncoderTrainer(model)
trainer.load_model(config.weight_path)

with torch.no_grad():
    trainer.model.eval()
    for method in ["f1", "roc", "percentile"]:
        eval_img = trainer.evaluate_image_level(test_loader, method=method)
        img_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_img.items() if isinstance(v, float)])
        img_info2 = ", ".join([f"{k}={v:2d}" for k, v in eval_img.items() if isinstance(v, int)])
        print(f" > Image-level: {img_info1} | {img_info2} ({method})")

with torch.no_grad():
    trainer.model.eval()
    eval_pix = trainer.evaluate_pixel_level(test_loader, percentile=95)
    pix_info = ", ".join([f"{k}={v:.3f}" for k, v in eval_pix.items() if isinstance(v, (int, float))])
    print(f" > Pixel-level: {pix_info}\n")

trainer.test(test_loader, show_image=True, output_dir=config.output_dir, 
    img_prefix=config.category, skip_normal=True, num_max=10)

trainer.test(test_loader, show_image=True, output_dir=config.output_dir, 
    img_prefix=config.category, skip_anomaly=True, num_max=10)
```

```python
import os
import torch
from dataloader import get_dataloaders
from model_autoencoder import AutoEncoder
from trainer import AutoEncoderTrainer
from main import get_config, show_model_info

category = "grid"
num_epochs = 20
output_dir = f"./results/{category}"

from model_manual import ManualEfficientAD
from trainer import ManualTrainer

config = get_config("manual", category, num_epochs, output_dir)
BACKBONE_DIR = "/home/namu/myspace/NAMU/project_2025/backbones"
teacher_backbone_path = os.path.join(BACKBONE_DIR, "efficientnet_b7_lukemelas-c5b4e57e.pth")
student_backbone_path = os.path.join(BACKBONE_DIR, "wide_resnet101_2-32ee1156.pth")
trainer = ManualTrainer(ManualEfficientAD(
    teacher_backbone_path=teacher_backbone_path,
    student_backbone_path=student_backbone_path,
    out_channels=128))
trainer.load_model(config.weight_path)
show_model_info(trainer.model)

train_loader, test_loader = get_dataloaders(config)

with torch.no_grad():
    trainer.model.eval()
    for method in ["f1", "roc", "percentile"]:
        eval_img = trainer.evaluate_image_level(test_loader, method=method)
        img_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_img.items() if isinstance(v, float)])
        img_info2 = ", ".join([f"{k}={v:2d}" for k, v in eval_img.items() if isinstance(v, int)])
        print(f" > Image-level: {img_info1} | {img_info2} ({method})")

with torch.no_grad():
    trainer.model.eval()
    eval_pix = trainer.evaluate_pixel_level(test_loader, percentile=95)
    pix_info = ", ".join([f"{k}={v:.3f}" for k, v in eval_pix.items() if isinstance(v, (int, float))])
    print(f" > Pixel-level: {pix_info}\n")

trainer.test(test_loader, show_image=True, output_dir=config.output_dir, 
    img_prefix=config.category, skip_normal=True, num_max=-1)

trainer.test(test_loader, show_image=True, output_dir=config.output_dir, 
    img_prefix=config.category, skip_anomaly=True, num_max=10)
```
