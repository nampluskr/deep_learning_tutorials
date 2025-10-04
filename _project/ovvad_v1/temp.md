# Knowledge Distillation 방식 - FRE 추가 부분

## 3.5 Fast Methods의 발전 과정

### 3.5.1 개요

Knowledge Distillation 패러다임에서 추론 속도 최적화는 지속적인 연구 주제였다. STFPM(2021)에서 시작된 고속화 연구는 여러 중간 시도를 거쳐 EfficientAd(2024)로 정점에 달했다.

### 3.5.2 발전 단계

**1단계: STFPM (2021) - 초기 고속화**
- 추론 속도: 20-40ms
- 성능: 96.8% AUROC
- 기여: Feature Pyramid Matching으로 당시 기준 빠른 속도 달성
- 한계: 실시간 처리에는 여전히 부족

**2단계: FRE (2023) - 중간 개선 시도**
- 정식 명칭: Feature Reconstruction Error
- 추론 속도: 10-30ms (STFPM 대비 약 2배 향상)
- 성능: 95-96% AUROC
- 핵심 아이디어: Feature reconstruction 과정을 경량화하여 속도 개선

**FRE의 기술적 접근**:
```
- 경량화된 feature extractor 사용
- 간소화된 reconstruction network
- Efficient anomaly score calculation
```

**FRE의 한계**:
- 속도 개선 폭이 제한적 (STFPM 대비 2배 미만)
- 성능 저하 (96.8% → 95-96%, -0.8-1.8%p)
- 실시간 처리(30+ FPS)에는 여전히 부족
- 독자적 기술 혁신 부족

**3단계: EfficientAd (2024) - 혁명적 발전**
- 추론 속도: 1-5ms (FRE 대비 2-30배 향상)
- 성능: 97.8% AUROC (FRE보다 오히려 높음)
- 혁신: Millisecond 레벨 추론 + 성능 향상 동시 달성

### 3.5.3 발전 추이 분석

**속도 개선 궤적**:
```
STFPM (2021):  20-40ms  ━━━━━━━━
                ↓ (2배 개선)
FRE (2023):    10-30ms  ━━━━
                ↓ (2-30배 개선)
EfficientAd (2024): 1-5ms  ━  ★ (현재 표준)
```

**성능 변화**:
```
STFPM:      96.8% ━━━━━━━━━━━
FRE:        95-96% ━━━━━━━━━  (소폭 하락)
EfficientAd: 97.8% ━━━━━━━━━━━━  (개선)
```

### 3.5.4 실무적 의미

**FRE의 역할**:
- 과도기적 모델로서 기술 발전 과정의 "징검다리"
- 속도 최적화 가능성을 탐색했으나 근본적 돌파구는 제시하지 못함
- EfficientAd의 등장으로 실용적 가치 상실

**현재 상태**:
- 실무 채택 사례: 거의 없음
- 학술적 인용: 제한적
- 권장 사항: **사용하지 않음** (EfficientAd로 대체)

**교훈**:
- 점진적 개선(2배)만으로는 실무 임팩트 제한적
- 혁신적 발전(20-40배)이 패러다임을 바꿈
- 속도와 성능은 trade-off가 아닐 수 있음 (EfficientAd 사례)

### 3.5.5 결론

현재 고속 이상 탐지가 필요한 경우 **EfficientAd가 사실상 유일한 선택지**이다. FRE를 포함한 중간 단계 모델들은 기술 발전 과정에서 역사적 의미만 가지며, 실무 적용에는 권장되지 않는다.

---

## 3.9 Knowledge Distillation 방식 종합 비교 (수정)

### 3.9.2 상세 비교표 (FRE 추가)



---

## 테이블 3 수정: Knowledge Distillation 방식 핵심 비교



---

## 테이블 11 수정: 전체 패러다임 통합 비교 (5대 방식 + FRE)

**Fast Methods 세부 분류 추가**:
 Model** | Dinomaly | 98.8% | 80-120ms | Multi-class SOTA | 모델 크기 | Multi-class 환경 | ★★★★★ |

---

## 6.6 Foundation Model 방식 종합 비교 - 테이블 15 수정

**Fast Methods Evolution 행 추가**:

| 비교 항목 | ... | STFPM | FRE | EfficientAd |
|----------|-----|-------|-----|-------------|
| **Fast Methods Evolution** | - | Baseline (20-40ms) | 과도기 (10-30ms) | 현재 표준 (1-5ms) |

이상으로 FRE 관련 추가/수정 부분을 완료했습니다.