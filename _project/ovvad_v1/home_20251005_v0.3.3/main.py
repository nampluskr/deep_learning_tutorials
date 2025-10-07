from train import train, train_models


if __name__ == "__main__":
    dataset_type, category = "mvtec", "tile"

    #############################################################
    # 1. Memory-based: PaDim(2020), PatchCore(2022), DFKDE(2022)
    #############################################################

    # train(dataset_type, category, "padim", num_epochs=1)
    # train(dataset_type, category, "patchcore", num_epochs=1)

    #############################################################
    # 2. Nomalizing Flow-based: CFlow(2021), FastFlow(2021), CSFlow(2021), UFlow(2022)
    #############################################################

    # train(dataset_type, category, "cflow-resnet18", num_epochs=3)
    # train(dataset_type, category, "cflow-resnet50", num_epochs=3)
    # train(dataset_type, category, "fastflow-resnet50", num_epochs=10)
    # train(dataset_type, category, "fastflow-cait", num_epochs=5)
    # train(dataset_type, category, "fastflow-deit", num_epochs=10)
    # train(dataset_type, category, "csflow", num_epochs=10)
    # train(dataset_type, category, "uflow-resnet50", num_epochs=10)
    # train(dataset_type, category, "uflow-mcait", num_epochs=10)

    #############################################################
    # 3. Knowledge Distillation: STFPM(2021), FRE(2023), Reverse Distillation(2022), EfficientAD(2024)
    #############################################################

    # train(dataset_type, category, "stfpm", num_epochs=50)
    # train(dataset_type, category, "fre", num_epochs=50)
    # train(dataset_type, category, "efficientad-small", num_epochs=10)
    # train(dataset_type, category, "efficientad-medium", num_epochs=10)
    # train(dataset_type, category, "reverse-distillation", num_epochs=50)

    #############################################################
    # 4. Reconstruction-based: GANomaly(2018), DRAEM(2021), DSR(2022)
    #############################################################

    # train(dataset_type, category, "autoencoder", num_epochs=50)
    # train(dataset_type, category, "draem", num_epochs=10)

    #############################################################
    # 5. Feature Adaptation: DFM(2019), CFA(2022)
    #############################################################

    # train(dataset_type, category, "dfm")
    # train(dataset_type, category, "cfa")

    train_models(dataset_type, category, model_list=["fre", "stfpm"])