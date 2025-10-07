import os
import torch

from dataloader import get_dataloaders
from registry import get_trainer, get_config


#####################################################################
# Global VARIABLES
#####################################################################

DATASET_DIR = "/mnt/d/datasets"
BACKBONE_DIR = "/mnt/d/backbones"
OUTPUT_DIR = "/mnt/d/outputs"
SEED = 42
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True


def set_globals(dataset_dir=None, backbone_dir=None, output_dir=None, 
               seed=None, num_workers=None, pin_memory=None, persistent_workers=None):
    global DATASET_DIR, BACKBONE_DIR, OUTPUT_DIR, SEED
    global NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS
    
    if dataset_dir is not None:
        DATASET_DIR = dataset_dir
    if backbone_dir is not None:
        BACKBONE_DIR = backbone_dir
    if output_dir is not None:
        OUTPUT_DIR = output_dir
    if seed is not None:
        SEED = seed

    if num_workers is not None:
        NUM_WORKERS = num_workers
    if pin_memory is not None:
        PIN_MEMORY = pin_memory
    if persistent_workers is not None:
        PERSISTENT_WORKERS = persistent_workers


def get_globals():
    return {
        "dataset_dir": DATASET_DIR,
        "backbone_dir": BACKBONE_DIR,
        "output_dir": OUTPUT_DIR,
        "seed": SEED,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
    }


def print_globals():
    config = get_globals()
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("="*70 + "\n")


def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(trainer):
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print()
    print(f" > Total params.:         {total_params:,}")
    print(f" > Trainable params.:     {trainable_params:,}")

    if trainer.optimizer is not None:
        optim_params = sum(p.numel() for group in trainer.optimizer.param_groups for p in group['params'])
        print(f" > Optimizer params.:     {optim_params:,}")


def train(dataset_type, category, model_type, num_epochs=None, batch_size=None, img_size=None, normalize=None):
    """Train a single model with automatic memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        # print_memory("Before Training")

    try:
        # Get configuration
        config = get_config(model_type)
        num_epochs = num_epochs or config["num_epochs"]
        img_size = img_size or config["img_size"]
        batch_size = batch_size or config["batch_size"]
        normalize = normalize if normalize is not None else config["normalize"]

        print(f"\n{'='*70}")
        print(f"  Training: {model_type} | {dataset_type}/{category}")
        print(f"  Epochs: {num_epochs}, Batch Size: {batch_size}, Image Size: {img_size}")
        print(f"{'='*70}\n")

        # Setup paths
        result_dir = os.path.join(OUTPUT_DIR, dataset_type, category, model_type)
        desc = f"{dataset_type}_{category}_{model_type}"
        weight_path=os.path.join(result_dir, f"model_{desc}_epochs-{num_epochs}.pth")

        # Set seed
        set_seed(seed=SEED)

        # Create dataloaders
        train_loader, test_loader = get_dataloaders(dataset_type, category,
            root_dir=os.path.join(DATASET_DIR, dataset_type),
            img_size=img_size,  batch_size=batch_size,  normalize=normalize,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
        # print_memory("After Dataloader Creation")

        # Create trainer
        trainer = get_trainer(model_type, backbone_dir=BACKBONE_DIR, dataset_dir=DATASET_DIR, img_size=img_size)
        count_parameters(trainer)
        print_memory("After Model Creation")

        # Train with validation
        trainer.fit(train_loader, num_epochs, valid_loader=test_loader, weight_path=weight_path)
        print_memory("After Training")

        # Save anomaly maps
        trainer.test(test_loader, result_dir=result_dir, desc=desc, normalize=normalize,
            skip_normal=True, num_max=20)
        trainer.test(test_loader, result_dir=result_dir, desc=desc, normalize=normalize,
            skip_anomaly=True, num_max=20)

    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR in {model_type}: {e}")
        print(f"{'!'*70}\n")
        import traceback
        traceback.print_exc()

    finally:
        if 'trainer' in locals():
            del trainer
        if 'train_loader' in locals():
            clear_dataloader(train_loader)
            del train_loader
        if 'test_loader' in locals():
            clear_dataloader(test_loader)
            del test_loader

        clear_memory(print_summary=True)


def clear_dataloader(dataloader):
    try:
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            del dataloader._iterator
        if hasattr(dataloader, 'worker_pids') and len(dataloader.worker_pids) > 0:
            dataloader._shutdown_workers()
        if hasattr(dataloader, '_persistent_workers') and dataloader._persistent_workers:
            dataloader._persistent_workers = False
    except Exception as e:
        print(f" > [Warning] Dataloader cleanup failed: {e}")


def clear_memory(print_summary=True, stage="After Cleanup"):
    import gc

    collected = gc.collect()
    if print_summary:
        print(f" > Python GC: Collected {collected} objects")

    if torch.cuda.is_available():
        if print_summary:
            before_allocated = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if print_summary:
            after_allocated = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3
            freed_allocated = before_allocated - after_allocated
            freed_reserved = before_reserved - after_reserved

            if freed_allocated > 0.01 or freed_reserved > 0.01:  # 10MB 이상만 표시
                print(f" > Freed: {freed_allocated:.2f} GB allocated, {freed_reserved:.2f} GB reserved")

            print_memory(stage)

        torch.cuda.reset_peak_memory_stats()


def print_memory(stage=""):
    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free_memory = total_memory - allocated
    usage = allocated / total_memory * 100

    stage_str = f"[{stage}]" if stage else ""
    print(f"\n{'GPU Memory ' + stage_str:-^70}")
    print(f"  Allocated: {allocated:.2f}GB ({usage:.1f}%) | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")
    print(f"  Free: {free_memory:.2f}GB / Total: {total_memory:.2f}GB")
    print(f"{'-'*70}")


def train_models(dataset_type, category, model_list, clear_memory_between=True):
    num_models = len(model_list)
    results = []
    for idx, model_type in enumerate(model_list, 1):
        print(f"{'='*70}")
        print(f"  Training: [{idx}/{num_models}] {model_type} | {dataset_type}/{category}")
        print(f"{'='*70}")

        try:
            train(dataset_type, category, model_type)
            results.append({"model": model_type, "status": "success"})

            if clear_memory_between and idx < num_models:
                print("\n > Additional memory cleanup between models...\n")
                clear_memory(print_summary=False)
                import time
                time.sleep(1)

        except Exception as e:
            print(f"\n[ERROR] Failed to train {model_type}: {e}")
            results.append({"model": model_type, "status": "failed", "error": str(e)})

    print(f"\n{'Training Summary':-^70}")
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    print(f"  Total Models:    {num_models}")
    print(f"  Successful:      {success_count}")
    print(f"  Failed:          {failed_count}")

    if failed_count > 0:
        print(f"\n  Failed Models:")
        for result in results:
            if result["status"] == "failed":
                print(f"    - {result['model']}: {result.get('error', 'Unknown error')}")
    print(f"{'='*70}")
    return results


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