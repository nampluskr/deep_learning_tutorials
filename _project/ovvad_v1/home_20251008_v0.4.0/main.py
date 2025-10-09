""" Available model types (from Anomlaib): version 1.0

1. Memory-based models:
  - [O] PaDim(2020): "padim"
  - [O] PatchCore(2022): "patchcore"
  - [X] DFKDE(2022)

2. Nomalizing Flow-based models:
  - [O] CFlow(2021): "cflow-resnet18", "cflow-resnet50"
  - [O] FastFlow(2021): "fastflow-resnet50", "fastflow-cait", "fastflow-deit"
  - [O] CSFlow(2021): "csflow"
  - [O] UFlow(2022): "uflow-resnet50", "uflow-mcait"

3. Knowledge Distillation models:
  - [O] STFPM(2021): "stfpm"
  - [O] FRE(2023): "fre"
  - [O] Reverse Distillation(2022): "reverse-distillation"
  - [O] EfficientAD(2024): "efficientad-small", "efficientad-medium"

4. Reconstruction-based models:
  - [X] Autoencoer: "autoencoder"
  - [X] GANomaly(2018)
  - [O] DRAEM(2021): "draem"
  - [X] DSR(2022)

5. Feature Adaptation models:
  - [O] DFM(2019): "dfm"
  - [O] CFA(2022): "cfa"
"""

from train import train, train_models, set_globals, print_globals

if __name__ == "__main__":

    #################################################################
    # Open datasets: MVTec / VisA / BTAD
    #################################################################

    set_globals(
        dataset_dir="/mnt/d/datasets",
        backbone_dir="/mnt/d/backbones",
        output_dir="/mnt/d/outputs",
        seed=42,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    print_globals()

    train_models(dataset_type="mvtec",
        categories=["wood", "grid", "tile"],
        models=["reverse-distillation", "fastflow-resnet50", "patchcore"]
    )

    train("mvtec", "wood", "stfpm", num_epochs=20)
    train("visa", "macaroni1", "stfpm", num_epochs=20)
    train("btad", "03", "stfpm", num_epochs=20)

    #################################################################    
    # Custon Dataset
    #################################################################
    
    set_globals(
        dataset_dir="/mnt/d/datasets/custom",
        backbone_dir="/mnt/d/backbones",
        output_dir="/mnt/d/outputs",
        seed=42,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    print_globals()
    
    train(["module1"], "tile", "stfpm", num_epochs=20)
    train(["module1"], ["grid", "tile"], "stfpm", num_epochs=20)
