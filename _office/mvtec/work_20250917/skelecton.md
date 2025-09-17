아래 코드는 **요구사항**에 맞게 `AnomalyTrainer` 를 다시 설계한 **스켈레톤**입니다.  

* `predict` 메서드는 삭제하고, `test` 와 `evaluate` 내부에서 `model.predict(batch)` 를 호출하도록 변경했습니다.  
* `train(train_loader)` 와 `validate(valid_loader)` 은 각각 **epoch‑단위** 학습·검증을 수행하고 `loss` / `psnr` / `ssim` (학습 진행을 모니터링하기 위한 metric) 을 반환합니다.  
* `fit(train_loader, num_epochs, valid_loader=None)` 은 내부에서 `train` 과 `validate` 를 순차적으로 호출해 **history** 를 누적합니다.  

---

## 1️⃣ `AnomalyModel` (변경 없음)

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class AnomalyModel(nn.Module):
    """
    이상 탐지 모델.
    - forward : (local_feat, global_feat) 반환
    - compute_anomaly_map / compute_anomaly_score : map·score 계산
    - predict : 배치 단위 예측을 수행하고 dict 로 반환
    """

    def __init__(self, backbone: nn.Module, proj_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone.out_channels, proj_dim),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    # 1) forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    # ------------------------------------------------------------------
    # 2) anomaly map
    # ------------------------------------------------------------------
    def compute_anomaly_map(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        ...

    # ------------------------------------------------------------------
    # 3) anomaly score (image‑level)
    # ------------------------------------------------------------------
    def compute_anomaly_score(
        self,
        anomaly_map: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        ...

    # ------------------------------------------------------------------
    # 4) 배치 예측 (predict) – test / evaluate 에서 호출
    # ------------------------------------------------------------------
    def predict(
        self,
        batch: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        배치에 대해 예측을 만든 뒤, 아래 키를 갖는 딕셔너리를 반환한다.

        Returns
        -------
        {
            "pred_score"   : torch.Tensor (B,)          # 이미지‑level score
            "anomaly_map"  : torch.Tensor (B, 1, H, W)   # 0~1 로 정규화된 map
            "binary_mask"  : torch.Tensor (B, 1, H, W)   # threshold 적용 mask
            "label"        : torch.Tensor (B,) (optional, GT)
        }
        """
        ...
```

---

## 2️⃣ `AnomalyTrainer` (요구사항 반영)

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class AnomalyTrainer:
    """
    AnomalyModel 전용 Trainer.
    - train(train_loader)                : epoch 단위 학습 → loss/psnr/ssim 반환
    - validate(valid_loader)             : 검증 → loss/psnr/ssim 반환
    - fit(train_loader, num_epochs, ...) : 내부에서 train/validate 를 호출하고 history 반환
    - test(test_loader, output_dir)      : model.predict 를 호출해 heatmap·mask·score 를 파일 저장
    - evaluate(test_loader)              : model.predict 로 전체 테스트 수행 후
                                          image‑wise / pixel‑wise AUROC·AUPR 반환
    """

    # ------------------------------------------------------------------
    # 1) 생성자
    # ------------------------------------------------------------------
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters
        ----------
        model      : nn.Module (AnomalyModel)
        optimizer  : torch.optim.Optimizer, 기본값 Adam(lr=1e‑4)
        loss_fn    : 손실 함수, 기본값 MSELoss
        device     : torch.device, 지정되지 않으면 CUDA 가 있으면 사용
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.optimizer = optimizer or optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = loss_fn or nn.MSELoss()

    # ------------------------------------------------------------------
    # 2) train (epoch‑level) → loss / psnr / ssim 반환
    # ------------------------------------------------------------------
    def train(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        한 epoch 동안 학습을 수행한다.
        반환값은 epoch 평균값을 담은 dict.

        Returns
        -------
        {
            "loss" : float,
            "psnr" : float,
            "ssim" : float,
        }
        """
        ...

    # ------------------------------------------------------------------
    # 3) validate (epoch‑level) → loss / psnr / ssim 반환
    # ------------------------------------------------------------------
    def validate(self, valid_loader: DataLoader) -> Dict[str, float]:
        """
        검증 데이터를 이용해 loss / psnr / ssim 평균값을 계산한다.
        Returns
        -------
        {
            "val_loss" : float,
            "val_psnr" : float,
            "val_ssim" : float,
        }
        """
        ...

    # ------------------------------------------------------------------
    # 4) fit (전체 학습 루프) → history 반환
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        valid_loader: Optional[DataLoader] = None,
        log_dir: str = "./logs",
        ckpt_dir: str = "./checkpoints",
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        전체 학습 루프.
        매 epoch 마다 `train()` 결과를 `history["train"]` 에,
        `valid_loader` 가 있으면 `validate()` 결과를 `history["val"]` 에 저장한다.

        Returns
        -------
        history : {
            "train" : [ {"loss":…, "psnr":…, "ssim":…},  # epoch 1
                        {"loss":…, "psnr":…, "ssim":…},  # epoch 2
                        ... ],
            "val"   : [ {"val_loss":…, "val_psnr":…, "val_ssim":…},  # epoch 1
                        {"val_loss":…, "val_psnr":…, "val_ssim":…},  # epoch 2
                        ... ]   # valid_loader 가 None 일 경우 빈 리스트
        }
        """
        ...

    # ------------------------------------------------------------------
    # 5) test (파일 저장) – 내부에서 model.predict 호출
    # ------------------------------------------------------------------
    def test(
        self,
        test_loader: DataLoader,
        output_dir: Path,
        ckpt_path: Optional[Path] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        체크포인트를 로드하고 `model.predict` 로 배치를 순회한다.
        각 배치에 대해
            - heatmap (PNG)
            - binary mask (PNG)
            - 이미지‑level score (JSON)
        를 `output_dir/predictions/` 에 저장한다.

        Returns
        -------
        {
            "pred_dir"   : Path,          # heatmap·mask·score 가 들어있는 폴더
            "num_images" : int,
            "summary"    : dict,          # test loss / (optional) AUROC·AUPR 등
        }
        """
        ...

    # ------------------------------------------------------------------
    # 6) evaluate (AUROC / AUPR) – 내부에서 model.predict 호출
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_loader: DataLoader,
        ckpt_path: Optional[Path] = None,
        pixel_level: bool = True,
        image_level: bool = True,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        전체 테스트 데이터를 `model.predict` 로 순회하면서
        - image‑wise AUROC / AUPR
        - pixel‑wise AUROC / AUPR (pixel_level=True 일 때)
        를 계산한다.

        Returns
        -------
        {
            "image_auroc" : float,
            "image_aupr"  : float,
            "pixel_auroc" : float,   # pixel_level=True 일 때만 포함
            "pixel_aupr"  : float,   # pixel_level=True 일 때만 포함
        }
        """
        ...

    # ------------------------------------------------------------------
    # (선택) checkpoint 저장/로드 헬퍼 – 구현은 자유
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, ckpt_dir: Path) -> Path:
        ...

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        ...
```

---

## 3️⃣ 흐름 예시 (스켈레톤 수준)

```python
# 1) 모델·Trainer 생성
backbone = ...                     # 예: torchvision.models.resnet18(pretrained=True)
model = AnomalyModel(backbone=backbone, proj_dim=128)

trainer = AnomalyTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    loss_fn=torch.nn.BCEWithLogitsLoss(),
)

# 2) 학습 (fit)
history = trainer.fit(
    train_loader=train_loader,
    num_epochs=30,
    valid_loader=val_loader,          # None 으로 두면 검증 없이 학습
    log_dir="./logs",
    ckpt_dir="./ckpts",
)

# 3) 테스트 파일 저장
test_res = trainer.test(
    test_loader=test_loader,
    output_dir=Path("./outputs"),
    ckpt_path=Path("./ckpts/best.ckpt"),
    threshold=0.5,
)

# 4) 최종 성능 평가 (AUROC / AUPR)
metrics = trainer.evaluate(
    test_loader=test_loader,
    ckpt_path=Path("./ckpts/best.ckpt"),
    pixel_level=True,
    image_level=True,
    threshold=0.5,
)

print("=== Evaluation ===")
print(metrics)   # {'image_auroc':0.962, 'image_aupr':0.945,
                 #  'pixel_auroc':0.874, 'pixel_aupr':0.714}
```

---

## 4️⃣ 핵심 정리

| 메서드 | 역할 | 반환값(예시) |
|-------|------|-------------|
| `train(train_loader)` | 한 epoch 학습 → `loss`, `psnr`, `ssim` 평균 | `{"loss":0.018, "psnr":28.4, "ssim":0.91}` |
| `validate(valid_loader)` | 한 epoch 검증 → 동일 metric 반환 | `{"val_loss":0.019, "val_psnr":27.9, "val_ssim":0.89}` |
| `fit(train_loader, num_epochs, valid_loader)` | `train`/`validate` 를 반복, `history` 반환 | `{"train":[...], "val":[...]}` |
| `test(test_loader, output_dir)` | `model.predict` 로 배치 순회 → heatmap·mask·score 파일 저장 | `{"pred_dir":Path(...), "num_images":200, "summary":{...}}` |
| `evaluate(test_loader)` | `model.predict` 로 전체 테스트 → image‑wise / pixel‑wise AUROC·AUPR 반환 | `{"image_auroc":0.962, "image_aupr":0.945, "pixel_auroc":0.874, "pixel_aupr":0.714}` |

위 스켈레톤을 기반으로 실제 구현(PSNR·SSIM 계산, AUROC·AUPR 로직, 파일 I/O 등)을 채워 넣으면, **학습‑검증‑테스트‑평가** 흐름이 명확히 구분된 깔끔한 이상 탐지 파이프라인을 만들 수 있습니다. 🚀
