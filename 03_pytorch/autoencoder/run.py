import subprocess
import os
from time import time
from datetime import timedelta

def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def run(file_list):
    """
    Run multiple training files
    
    Args:
        file_list: List of python training files
    """
    total_runs = len(file_list)
    total_start = time()
    
    print(f"Starting {total_runs} training files...")
    print(f"Files: {file_list}")
    print("=" * 80)
    
    for file_idx, filename in enumerate(file_list):
        current_run = file_idx + 1
        start = time()
        
        print(f"\n>> Run[{current_run}/{total_runs}] File: {filename}")
        
        # Build command
        command = ["python", filename]
        
        try:
            # Run the training script
            result = subprocess.run(command, check=True, capture_output=False)
            status = "✓ SUCCESS"
        except subprocess.CalledProcessError as e:
            status = f"✗ FAILED (exit code: {e.returncode})"
        except FileNotFoundError:
            status = f"✗ FAILED (file not found: {filename})"
        
        elapsed_time = format_time(time() - start)
        print(f">> Status: {status} | Time: {elapsed_time}")
        print("-" * 80)
    
    total_elapsed = format_time(time() - total_start)
    print(f"\n🎉 Total execution time: {total_elapsed}")
    print("=" * 80)
    print("All training runs completed!")
    print("\n📋 Usage:")
    print("   python run.py")

if __name__ == "__main__":
    # Training files for OLED anomaly detection
    file_list = [
        "main_vanilla_ae.py",           # Vanilla AutoEncoder
        "main_improved_ae.py",          # Improved AutoEncoder with skip connections
        "main_vae.py",                  # Variational AutoEncoder
        "main_padim.py",                # PaDiM (Patch Distribution Modeling)
        "main_patchcore.py",            # PatchCore
        "main_fastflow.py",             # FastFlow
    ]
    
    # Check which files actually exist
    existing_files = []
    for filename in file_list:
        if os.path.exists(filename):
            existing_files.append(filename)
        else:
            print(f"Warning: File not found: {filename}")
    
    if not existing_files:
        print("Error: No training files found!")
        exit(1)
    
    # Run all existing files
    run(existing_files)