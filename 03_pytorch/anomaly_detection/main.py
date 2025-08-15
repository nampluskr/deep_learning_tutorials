"""
메인 실행 파이프라인 및 실험 관리
"""

import os
import numpy as np
import torch

# 각 모듈에서 필요한 함수들 import
from mvtec import get_transforms, get_dataloaders
from vanila_ae import VanillaAutoEncoder  
from train import get_device, train_model, evaluate_model
from utils import (
    visualize_training_history,
    visualize_reconstruction_examples, 
    visualize_score_distribution,
    create_results_summary_table,
    plot_category_comparison
)


def main_pipeline(data_dir, categories=None, num_epochs=50, batch_size=32, 
                 latent_dim=512, img_size=256, normalize=False,
                 save_models=True, save_results=True):
    """
    메인 실험 파이프라인
    
    Args:
        data_dir: MVTec 데이터셋 루트 디렉토리
        categories: 실험할 카테고리 리스트 (None이면 전체)
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        latent_dim: 잠재 공간 차원
        img_size: 이미지 크기
        normalize: ImageNet 정규화 사용 여부
        save_models: 모델 저장 여부
        save_results: 결과 저장 여부
    
    Returns:
        all_category_results: 카테고리별 결과 리스트
        results_df: 결과 DataFrame
    """
    
    # 설정 및 초기화
    device = get_device(seed=42)
    train_transform, test_transform = get_transforms(img_size=img_size, normalize=normalize)
    
    # MVTec 카테고리 정의 (전체 또는 선택)
    if categories is None:
        categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
    
    all_category_results = []
    
    print(f"Starting MVTec Anomaly Detection Evaluation")
    print(f"Total categories: {len(categories)}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"Latent dimension: {latent_dim}, Image size: {img_size}")
    print(f"Normalize: {normalize}")
    print("="*60)
    
    # 카테고리별 실험
    for i, category in enumerate(categories):
        print(f"\n🔍 Processing Category {i+1}/{len(categories)}: {category.upper()}")
        print("-" * 50)
        
        try:
            # 1. 데이터 로더 생성
            print("Creating dataloaders...")
            train_loader, valid_loader, test_loader = get_dataloaders(
                data_dir, category, batch_size,
                train_transform=train_transform, 
                test_transform=test_transform
            )
            
            # 2. 모델 초기화
            print("Initializing model...")
            model = VanillaAutoEncoder(latent_dim=latent_dim).to(device)
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = total_params * 4 / (1024 * 1024)  # float32 기준
            print(f"Model parameters: {total_params:,}")
            print(f"Model size: {model_size_mb:.1f} MB")
            
            # 3. 모델 훈련
            print("Training model...")
            trained_model, history = train_model(
                model, train_loader, valid_loader, device, num_epochs=num_epochs
            )
            
            # 4. 테스트 평가
            print("Evaluating model...")
            results, detailed_results = evaluate_model(trained_model, test_loader, device)
            
            # 5. 결과 저장
            results['Category'] = category
            all_category_results.append(results)
            
            # 6. 시각화
            print("Creating visualizations...")
            try:
                visualize_training_history(history, category)
                visualize_reconstruction_examples(detailed_results, category)
                visualize_score_distribution(detailed_results, category)
            except Exception as viz_error:
                print(f"Warning: Visualization error for {category}: {viz_error}")
            
            # 7. 결과 출력
            print(f"\n📊 Results for {category}:")
            print(f"  AUROC: {results['AUROC']:.4f}")
            print(f"  AUPR:  {results['AUPR']:.4f}")
            print(f"  F1:    {results['F1-Score']:.4f}")
            print(f"  Accuracy: {results['Accuracy']:.4f}")
            print(f"  Threshold: {results['Threshold']:.6f}")
            
            # 8. 모델 저장 (선택사항)
            if save_models:
                os.makedirs("models", exist_ok=True)
                model_path = f"models/vanilla_ae_{category}.pth"
                torch.save(trained_model.state_dict(), model_path)
                print(f"Model saved: {model_path}")
            
        except Exception as e:
            print(f"❌ Error processing {category}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 실패한 카테고리에 대해 더미 결과 추가
            dummy_results = {
                'Category': category,
                'AUROC': 0.0,
                'AUPR': 0.0,
                'Accuracy': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1-Score': 0.0,
                'Threshold': 0.0,
                'Normal_Score_Mean': 0.0,
                'Normal_Score_Std': 0.0,
                'Anomaly_Score_Mean': 0.0
            }
            all_category_results.append(dummy_results)
            continue
    
    # 전체 결과 분석 및 시각화
    print("\n" + "="*60)
    print("📈 FINAL RESULTS SUMMARY")
    print("="*60)
    
    # 결과 테이블 생성
    try:
        results_df = create_results_summary_table(all_category_results)
    except Exception as e:
        print(f"Error creating summary table: {e}")
        import pandas as pd
        results_df = pd.DataFrame(all_category_results)
    
    # 카테고리별 성능 비교 시각화
    try:
        plot_category_comparison(all_category_results)
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
    
    # 결과 저장
    if save_results:
        try:
            results_df.to_csv("mvtec_vanilla_ae_results.csv", index=False)
            print("Results saved to 'mvtec_vanilla_ae_results.csv'")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # 최고/최저 성능 카테고리 분석
    if len(all_category_results) > 0:
        # 실패하지 않은 결과들만 필터링
        valid_results = [r for r in all_category_results if r['AUROC'] > 0]
        
        if len(valid_results) > 0:
            auroc_scores = [r['AUROC'] for r in valid_results]
            best_auroc_idx = np.argmax(auroc_scores)
            worst_auroc_idx = np.argmin(auroc_scores)
            
            print(f"\n🏆 Best performing category: {valid_results[best_auroc_idx]['Category']} "
                  f"(AUROC: {valid_results[best_auroc_idx]['AUROC']:.4f})")
            print(f"📉 Challenging category: {valid_results[worst_auroc_idx]['Category']} "
                  f"(AUROC: {valid_results[worst_auroc_idx]['AUROC']:.4f})")
            
            overall_mean_auroc = np.mean(auroc_scores)
            print(f"📊 Overall mean AUROC: {overall_mean_auroc:.4f}")
            
            # 성능 분포 분석
            excellent_count = sum(1 for score in auroc_scores if score >= 0.9)
            good_count = sum(1 for score in auroc_scores if 0.8 <= score < 0.9)
            fair_count = sum(1 for score in auroc_scores if 0.7 <= score < 0.8)
            poor_count = sum(1 for score in auroc_scores if score < 0.7)
            
            print(f"\n📈 Performance Distribution:")
            print(f"  Excellent (≥0.9): {excellent_count}/{len(valid_results)} categories")
            print(f"  Good (0.8-0.9): {good_count}/{len(valid_results)} categories")
            print(f"  Fair (0.7-0.8): {fair_count}/{len(valid_results)} categories")
            print(f"  Poor (<0.7): {poor_count}/{len(valid_results)} categories")
    
    return all_category_results, results_df


def quick_test(data_dir, test_categories=None, epochs=5):
    """
    빠른 테스트 실행 (개발 및 디버깅용)
    
    Args:
        data_dir: MVTec 데이터셋 경로
        test_categories: 테스트할 카테고리들 (None이면 ['bottle'])
        epochs: 훈련 에포크 수
    """
    if test_categories is None:
        test_categories = ['bottle']
    
    print("🚀 Quick Test Mode")
    print(f"Testing categories: {test_categories}")
    print(f"Epochs: {epochs}")
    
    results, df = main_pipeline(
        data_dir=data_dir,
        categories=test_categories,
        num_epochs=epochs,
        batch_size=16,  # 작은 배치 크기
        img_size=256,
        normalize=False,  # 정규화 사용 안함
        save_models=False,
        save_results=False
    )
    
    print("\n✅ Quick test completed!")
    return results, df


def test_experiment(data_dir):
    """
    중간 크기 테스트 실험 (3개 카테고리)
    
    Args:
        data_dir: MVTec 데이터셋 경로
    """
    print("🧪 Test Experiment")
    print("Testing with 3 categories...")
    
    test_categories = ['bottle', 'grid', 'tile']
    
    results, df = main_pipeline(
        data_dir=data_dir,
        categories=test_categories,
        num_epochs=15,
        batch_size=24,
        img_size=256,
        normalize=False,
        save_models=True,
        save_results=True
    )
    
    print("\n🎯 Test experiment completed!")
    return results, df


def full_experiment(data_dir):
    """
    전체 MVTec 카테고리에 대한 완전한 실험
    
    Args:
        data_dir: MVTec 데이터셋 경로
    """
    print("🔬 Full MVTec Experiment")
    print("This will take several hours...")
    
    results, df = main_pipeline(
        data_dir=data_dir,
        categories=None,  # 전체 카테고리
        num_epochs=50,
        batch_size=32,
        latent_dim=512,
        img_size=256,
        normalize=False,
        save_models=True,
        save_results=True
    )
    
    print("\n🎉 Full experiment completed!")
    return results, df


def experiment_with_normalization(data_dir):
    """
    ImageNet 정규화를 사용한 실험
    
    Args:
        data_dir: MVTec 데이터셋 경로
    """
    print("🔍 Experiment with ImageNet Normalization")
    
    test_categories = ['bottle', 'grid', 'tile']
    
    results, df = main_pipeline(
        data_dir=data_dir,
        categories=test_categories,
        num_epochs=20,
        batch_size=32,
        img_size=256,
        normalize=True,  # ImageNet 정규화 사용
        save_models=True,
        save_results=True
    )
    
    print("\n📊 Normalization experiment completed!")
    return results, df


# 실행 부분
if __name__ == "__main__":
    # 데이터셋 경로 설정 (실제 경로로 변경 필요)
    DATA_DIR = "/path/to/mvtec_anomaly_detection"
    
    # 실행 모드 선택
    RUN_MODE = "quick"  # "quick", "test", "full", "norm" 중 선택
    
    print("="*60)
    print("MVTec Anomaly Detection with Vanilla AutoEncoder")
    print("="*60)
    
    # GPU 메모리 확인
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        if RUN_MODE == "quick":
            # 빠른 테스트 (1개 카테고리, 5 에포크)
            results, df = quick_test(DATA_DIR, ['bottle'], epochs=5)
            
        elif RUN_MODE == "test":
            # 중간 테스트 (3개 카테고리, 15 에포크)
            results, df = test_experiment(DATA_DIR)
            
        elif RUN_MODE == "full":
            # 전체 실험 (모든 카테고리, 50 에포크)
            results, df = full_experiment(DATA_DIR)
            
        elif RUN_MODE == "norm":
            # 정규화 실험 (3개 카테고리, ImageNet 정규화 사용)
            results, df = experiment_with_normalization(DATA_DIR)
            
        else:
            print(f"Unknown run mode: {RUN_MODE}")
            print("Available modes: 'quick', 'test', 'full', 'norm'")
    
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎯 Experiment completed!")
    print("Check the generated visualizations and results.")
    print("="*60)