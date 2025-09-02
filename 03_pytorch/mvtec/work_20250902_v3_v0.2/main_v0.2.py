import json
import time
import os
from config import build_config, get_available_types, print_config
from factory import run_experiment, get_available_combinations

# ===================================================================
# EXPERIMENT CONFIGURATION
# ===================================================================

MODELS = ["unet_ae", "stfpm", "patchcore", "padim", "fastflow", "draem"]
DATASETS = ["mvtec", "btad", "visa"]

TEST_MODE = False
TEST_CATEGORIES = {
    "mvtec": ["bottle", "cable"],
    "btad": ["01"],
    "visa": ["candle"],
} if TEST_MODE else None

# ===================================================================
# SINGLE EXPERIMENT RUNNER
# ===================================================================

def run_single_experiment(dataset_type, model_type, custom_overrides=None):
    config = build_config(dataset_type, model_type, overrides=custom_overrides)

    # Limit categories for testing
    if TEST_CATEGORIES and dataset_type in TEST_CATEGORIES:
        config.categories = TEST_CATEGORIES[dataset_type]

    start_time = time.time()

    try:
        result = run_experiment(dataset_type, model_type, config)
        elapsed_time = time.time() - start_time

        print(f"[OK] {dataset_type} + {model_type} ({elapsed_time:.1f}s) | "
              f"scores_shape={result['scores'].shape}, "
              f"train_batch={result['config_summary']['train_batch']}")

        result["elapsed_time"] = elapsed_time
        result["status"] = "success"
        return result

    except Exception as error:
        elapsed_time = time.time() - start_time
        print(f"[FAIL] {dataset_type} + {model_type} ({elapsed_time:.1f}s) -> {type(error).__name__}: {error}")
        return {
            "status": "failed",
            "error": str(error),
            "error_type": type(error).__name__,
            "elapsed_time": elapsed_time
        }

# ===================================================================
# GRID EXPERIMENT RUNNER
# ===================================================================

def run_grid_experiments():
    available_datasets, available_models = get_available_combinations()

    print("=" * 60)
    print("STARTING GRID EXPERIMENTS")
    print("=" * 60)
    print(f"Available datasets: {available_datasets}")
    print(f"Available models: {available_models}")
    print(f"Running models: {MODELS}")
    print(f"Running datasets: {DATASETS}")
    print(f"Test mode: {TEST_MODE}")

    experiment_results = {}
    total_experiments = len(DATASETS) * len(MODELS)
    completed_experiments = 0

    start_time = time.time()

    for dataset_type in DATASETS:
        if dataset_type not in available_datasets:
            print(f"[SKIP] Dataset '{dataset_type}' not available")
            continue

        for model_type in MODELS:
            if model_type not in available_models:
                print(f"[SKIP] Model '{model_type}' not available")
                continue

            completed_experiments += 1
            experiment_id = f"{dataset_type}_{model_type}"

            print(f"\n[{completed_experiments}/{total_experiments}] {experiment_id}")

            result = run_single_experiment(dataset_type, model_type)
            experiment_results[experiment_id] = result

    elapsed_total_time = time.time() - start_time
    success_count = sum(1 for result in experiment_results.values() if result["status"] == "success")

    print(f"\n{'='*60}")
    print("GRID EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_total_time:.1f}s")
    print(f"Success: {success_count}/{len(experiment_results)}")
    print(f"{'='*60}")

    return experiment_results

# ===================================================================
# SPECIFIC EXPERIMENTS RUNNER
# ===================================================================

def run_specific_experiments(experiment_list, custom_configs=None):
    specific_results = {}
    custom_configs = custom_configs or {}

    print("=" * 60)
    print("RUNNING SPECIFIC EXPERIMENTS")
    print("=" * 60)

    for i, (dataset_type, model_type) in enumerate(experiment_list, 1):
        experiment_id = f"{dataset_type}_{model_type}"
        print(f"\n[{i}/{len(experiment_list)}] {experiment_id}")

        # Get custom config for this experiment if exists
        custom_overrides = custom_configs.get(experiment_id, None)

        result = run_single_experiment(dataset_type, model_type, custom_overrides)
        specific_results[experiment_id] = result

    return specific_results

# ===================================================================
# RESULTS MANAGEMENT
# ===================================================================

def save_results(experiment_results, filename="experiment_results.json"):

    # Convert numpy arrays to lists for JSON serialization
    def convert_arrays(obj):
        if hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_arrays(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        else:
            return obj

    json_results = convert_arrays(experiment_results)

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(json_results, file, indent=2, default=str)
    print(f"Results saved to {filename}")


def print_experiment_summary(experiment_results):
    if not experiment_results:
        print("No experiment results to summarize.")
        return

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    successful = []
    failed = []

    for experiment_id, result in experiment_results.items():
        if result["status"] == "success":
            successful.append((experiment_id, result["elapsed_time"]))
        else:
            failed.append((experiment_id, result.get("error_type", "Unknown")))

    print(f"Total experiments: {len(experiment_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSUCCESSFUL EXPERIMENTS:")
        for exp_id, time_taken in sorted(successful, key=lambda x: x[1]):
            print(f"  {exp_id}: {time_taken:.1f}s")

    if failed:
        print("\nFAILED EXPERIMENTS:")
        for exp_id, error_type in failed:
            print(f"  {exp_id}: {error_type}")

    print("=" * 60)

# ===================================================================
# CONFIGURATION UTILITIES
# ===================================================================

def print_available_configurations():
    available_datasets, available_models = get_available_combinations()
    config_datasets, config_models = get_available_types()

    print("=" * 60)
    print("AVAILABLE CONFIGURATIONS")
    print("=" * 60)
    print(f"Factory datasets: {available_datasets}")
    print(f"Factory models: {available_models}")
    print(f"Config datasets: {config_datasets}")
    print(f"Config models: {config_models}")
    print("=" * 60)

def debug_configurations(combinations):
    """Debug specific configurations by printing them"""
    print("=" * 60)
    print("CONFIGURATION DEBUG")
    print("=" * 60)

    for dataset_type, model_type in combinations:
        print_config(dataset_type, model_type)

# ===================================================================
# MAIN EXECUTION FUNCTION
# ===================================================================

def main():

    # ---- Mode 1: Single experiment with custom config ----
    if False:
        print("Running single experiment with custom config...")
        custom_overrides = {
            "train_batch_size": 8,
            "trainer_params": {"epochs": 50, "lr": 5e-4}
        }
        result = run_single_experiment("mvtec", "unet_ae", custom_overrides)
        save_results({"mvtec_unet_ae": result}, "single_experiment.json")
        return

    # ---- Mode 2: Configuration debugging ----
    if False:
        print("Debugging configurations...")
        debug_combinations = [
            ("mvtec", "patchcore"),
            ("btad", "fastflow"),
            ("visa", "stfpm")
        ]
        debug_configurations(debug_combinations)
        return

    # ---- Mode 3: Print available configurations ----
    if False:
        print_available_configurations()
        return

    # ---- Mode 4: Run specific experiments ----
    if False:
        specific_experiments = [
            ("mvtec", "unet_ae"),
            ("mvtec", "patchcore"),
            ("btad", "stfpm"),
            ("visa", "fastflow"),
        ]

        # Optional: custom configs for specific experiments
        custom_configs = {
            "mvtec_unet_ae": {
                "train_batch_size": 32,
                "trainer_params": {"epochs": 150, "lr": 2e-3}
            },
            "btad_stfpm": {
                "train_batch_size": 8,
                "trainer_params": {"epochs": 80}
            }
        }

        specific_results = run_specific_experiments(specific_experiments, custom_configs)
        save_results(specific_results, "specific_results.json")
        print_experiment_summary(specific_results)
        return

    # ---- Mode 5: Run full grid (default) ----
    experiment_results = run_grid_experiments()
    save_results(experiment_results, "grid_results.json")
    print_experiment_summary(experiment_results)

if __name__ == "__main__":
    main()