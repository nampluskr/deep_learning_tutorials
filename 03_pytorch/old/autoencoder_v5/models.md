## 📊 SOTA Anomaly Detection 모델 분석

### **1. 모델 패러다임 분류**

| 카테고리 | 모델들 | 핵심 원리 | 복잡도 |
|----------|--------|-----------|--------|
| **Reconstruction-based** | VanillaAE, UnetAE, VAE, β-VAE | 재구성 오차 기반 | 중간 |
| **Memory-based** | PatchCore, SPADE | Feature memory bank + KNN | 높음 |
| **Flow-based** | FastFlow, DifferNet | Normalizing flow | 매우 높음 |
| **Distillation-based** | EfficientAD, Student-Teacher | Knowledge distillation | 높음 |
| **Statistical-based** | PaDiM, STFPM | 통계적 모델링 | 중간 |
| **Hybrid** | EfficientAD (AE+Distillation) | 다중 접근법 조합 | 매우 높음 |

### **2. 코드 복잡도 및 의존성 분석**

| 모델 | 예상 코드 길이 | 외부 의존성 | 공통 코드 재사용 |
|------|---------------|-------------|-----------------|
| **Autoencoder 계열** | 400-600줄 | PyTorch 기본 | Building blocks 공유 |
| **PatchCore** | 300-400줄 | FAISS, sklearn | Feature extractor 공유 |
| **PaDiM** | 200-300줄 | sklearn | Feature extractor 공유 |
| **FastFlow** | 400-500줄 | FrEIA, normflows | Complex flow 구조 |
| **EfficientAD** | 300-400줄 | PyTorch 기본 | AE + Feature extractor |

## 🎯 **권장 파일 구조 (Option A: 방법론별 분리)**

### **📁 models/ 패키지 구조**
```
models/
├── __init__.py                    # Unified factory & imports
├── base/
│   ├── __init__.py
│   ├── building_blocks.py         # ConvBlock, DeconvBlock 등
│   ├── feature_extractors.py      # Pretrained backbones (ResNet, EfficientNet)
│   └── utils.py                   # 공통 유틸리티 함수들
│
├── reconstruction/
│   ├── __init__.py
│   ├── vanilla.py                 # VanillaAE, UnetAE
│   ├── variational.py             # VAE, β-VAE, WAE
│   └── pretrained.py              # ResNet/VGG encoder 기반 AE
│
├── memory_based/
│   ├── __init__.py
│   ├── patchcore.py               # PatchCore 구현
│   ├── spade.py                   # SPADE 구현  
│   └── memory_utils.py            # Memory bank, KNN utilities
│
├── flow_based/
│   ├── __init__.py
│   ├── fastflow.py                # FastFlow 구현
│   ├── differnet.py               # DifferNet 구현
│   └── flow_utils.py              # Normalizing flow utilities
│
├── statistical/
│   ├── __init__.py
│   ├── padim.py                   # PaDiM 구현
│   ├── stfpm.py                   # STFPM 구현
│   └── stat_utils.py              # 통계 모델링 utilities
│
└── distillation/
    ├── __init__.py
    ├── efficientad.py             # EfficientAD 구현
    ├── student_teacher.py         # Student-Teacher 구현
    └── distill_utils.py           # Knowledge distillation utilities
```

## 🔧 **Unified Factory 구현 전략**

### **models/__init__.py**
```python
# All model imports
from .reconstruction import get_reconstruction_model
from .memory_based import get_memory_model  
from .flow_based import get_flow_model
from .statistical import get_statistical_model
from .distillation import get_distillation_model

def get_model(model_type, **kwargs):
    """Unified model factory for all anomaly detection methods"""
    
    model_categories = {
        # Reconstruction-based
        'vanilla_ae': ('reconstruction', 'vanilla_ae'),
        'unet_ae': ('reconstruction', 'unet_ae'), 
        'vae': ('reconstruction', 'vae'),
        'beta_vae': ('reconstruction', 'beta_vae'),
        'resnet_ae': ('reconstruction', 'resnet_ae'),
        
        # Memory-based
        'patchcore': ('memory_based', 'patchcore'),
        'spade': ('memory_based', 'spade'),
        
        # Flow-based  
        'fastflow': ('flow_based', 'fastflow'),
        'differnet': ('flow_based', 'differnet'),
        
        # Statistical
        'padim': ('statistical', 'padim'),
        'stfpm': ('statistical', 'stfpm'),
        
        # Distillation-based
        'efficientad': ('distillation', 'efficientad'),
        'student_teacher': ('distillation', 'student_teacher'),
    }
    
    if model_type not in model_categories:
        available = ', '.join(model_categories.keys())
        raise ValueError(f"Unknown model: {model_type}. Available: {available}")
    
    category, model_name = model_categories[model_type]
    
    if category == 'reconstruction':
        return get_reconstruction_model(model_name, **kwargs)
    elif category == 'memory_based':
        return get_memory_model(model_name, **kwargs)
    elif category == 'flow_based':
        return get_flow_model(model_name, **kwargs)
    elif category == 'statistical':
        return get_statistical_model(model_name, **kwargs)
    elif category == 'distillation':
        return get_distillation_model(model_name, **kwargs)
```

## 🎯 **Alternative Option B: 간소화된 구조**

```
models/
├── __init__.py              # Unified factory
├── base.py                  # 공통 building blocks & utilities
├── classical.py             # Autoencoder 계열 (VAE 포함)
└── sota.py                  # 최신 SOTA 모델들 (PatchCore, PaDiM, FastFlow, EfficientAD)
```

### **Option B의 장단점**
**장점**: 파일 수 적음, 관리 단순
**단점**: sota.py가 매우 크고 복잡해짐 (1000+줄), 서로 다른 패러다임 혼재

## 📋 **모델별 주요 구현 요소**

### **PatchCore**
- Feature extraction (ResNet 등)
- Memory bank 관리
- KNN search (FAISS)
- Patch-level scoring

### **PaDiM**  
- Pretrained feature extraction
- Multivariate Gaussian modeling
- Mahalanobis distance calculation
- Multi-scale feature aggregation

### **FastFlow**
- Normalizing flow architecture
- Coupling layers
- Log-likelihood calculation
- Flow inversion

### **EfficientAD**
- Student-Teacher networks
- Autoencoder component
- Multi-loss training
- Feature alignment

## 🎯 **최종 권장사항: Option A (방법론별 분리)**

### **이유:**

1. **논리적 분리**: 각 방법론별로 완전히 다른 접근법
2. **독립적 개발**: 팀원들이 각 방법론을 독립적으로 개발 가능
3. **의존성 관리**: Flow 기반은 복잡한 외부 라이브러리 필요
4. **코드 재사용**: base/ 폴더의 공통 요소 활용
5. **확장성**: 새로운 SOTA 모델 추가시 해당 카테고리에만 추가
6. **테스트**: 방법론별 독립적 테스트
7. **문서화**: 각 방법론별 별도 문서 가능

### **사용 예시:**
```python
# 간단한 통합 사용
from models import get_model
model = get_model('patchcore', backbone='resnet50')
model = get_model('fastflow', input_size=256)

# 직접 import도 가능  
from models.memory_based import PatchCore
model = PatchCore(backbone='resnet50')

# 카테고리별 모델 리스트
from models import list_models_by_category
print(list_models_by_category('memory_based'))  # ['patchcore', 'spade']
```

**이 구조로 진행하시겠습니까?**