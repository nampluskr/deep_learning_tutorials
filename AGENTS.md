# deep_learning_tutorials — Codex 지침

이 워크스페이스는 WSL + Anaconda + PyTorch + CUDA + VS Code + Codex 기반으로 딥러닝 tutorial 예제를 작성·실행·검증하기 위한 공간이다.
주요 대상은 Computer Vision tutorial, PyTorch 예제, 모델 학습·추론 실험이다.

---

## AI 역할 범위

**MUST**
- PyTorch 기반 예제 코드 작성·수정·디버깅을 보조한다.
- Computer Vision tutorial, 실험 코드, 실행 스크립트, 테스트 코드 작성을 보조한다.
- 작업 폴더의 `SPEC.md`, `ROADMAP.md`, `README.md`를 기준으로 범위를 판단한다.
- 코드 수정 후 가능한 경우 smoke test, import test, 단일 batch forward pass, 짧은 실행 검증 중 하나를 제안하거나 수행한다.

**MUST NOT**
- 사용자 승인 없이 파일·폴더를 생성, 이동, 삭제하지 않는다.
- 사용자 승인 없이 워크스페이스 구조나 운영 원칙을 변경하지 않는다.
- 작업 범위를 `SPEC.md`에 정의된 목적 밖으로 임의 확장하지 않는다.

---

## 협업 규칙

### 파일 생성·수정 절차

**MUST**
- 파일 생성·수정 전 대상 경로와 내용을 먼저 제안한다.
- 사용자 승인 후 실행한다.
- 여러 파일을 수정할 경우 수정 범위를 먼저 설명하고 진행 여부를 확인한다.

**MUST NOT**
- 사용자의 명시적 승인 없이 파일·폴더를 생성하지 않는다.
- "검토해 주세요"는 생성·수정 승인으로 해석하지 않는다.
- 구조 변경, 폴더 생성, 파일 이동, 삭제는 반드시 사용자 확인 후 실행한다.

### 응답 스타일

**MUST**
- 간결하게 응답한다.
- 기술 용어, 패키지명, 파일명, 경로는 영어 원문 그대로 사용한다.
- 코드 주석은 WHY가 명확할 때만 작성한다.

**MUST NOT**
- 이모지를 사용하지 않는다.
- 불필요한 칭찬·감탄 표현을 사용하지 않는다.

### 편집 환경 표준

- 인코딩: UTF-8
- 줄 끝: LF
- Markdown/YAML 들여쓰기: 2-space
- Python 들여쓰기: 4-space

---

## 실행 환경

### 기본 환경

- OS 환경: WSL
- Python 환경: Anaconda
- 기본 conda 환경 이름: `pytorch_env`
- Deep Learning framework: PyTorch
- IDE: VS Code
- AI coding assistant: Codex

작업 실행 전 기본적으로 아래 환경을 활성화한다고 가정한다.

```bash
conda activate pytorch_env
```

### CUDA 기준

이 워크스페이스의 tutorial 코드는 CUDA 사용을 공통 전제로 한다.

**MUST**
- PyTorch 코드는 기본 device로 `torch.device("cuda")`를 사용한다.
- 실행 초기에 `torch.cuda.is_available()`를 확인한다.
- CUDA가 사용할 수 없는 경우 명확한 오류를 발생시킨다.

**MUST NOT**
- 사용자 요청 없이 CPU fallback을 기본 제공하지 않는다.

권장 device 확인 코드는 다음과 같다.

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this tutorial.")

DEVICE = torch.device("cuda")
```

### 외부 리소스 경로

이 워크스페이스 외부에 다음 폴더가 존재한다고 가정한다.

```text
datasets/   -> DATASET_DIR
backbones/  -> BACKBONE_DIR
```

코드에서는 파일 상단에 다음 기준으로 경로를 정의한다.

```python
import os
from pathlib import Path

DATASET_DIR = Path(os.getenv("DATASET_DIR", "/path/to/datasets"))
BACKBONE_DIR = Path(os.getenv("BACKBONE_DIR", "/path/to/backbones"))
```

**MUST NOT**
- 대용량 dataset, pretrained backbone, checkpoint를 저장소 내부에 커밋 대상으로 추가하지 않는다.

---

## 워크스페이스 구조

루트 구조는 다음 기준을 따른다.

```text
deep_learning_tutorials/
├── AGENTS.md
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── .vscode/
│   └── settings.json
├── _workspace/
│   ├── docs/
│   │   ├── myst.yml
│   │   └── contents/
│   └── sessions/
├── works/
│   ├── 01_sandbox/
│   ├── 02_experiments/
│   └── 03_projects/
└── archive/
    ├── 01_sandbox/
    ├── 02_experiments/
    └── 03_projects/
```

### 루트 폴더 역할

| 경로 | 역할 |
|------|------|
| `_workspace/docs/` | 전역 참고 문서, 환경 설정 메모, 공통 실행 규칙 |
| `_workspace/sessions/` | 특정 작업에 속하지 않는 세션 핸드오프 |
| `works/01_sandbox/` | 빠른 확인, API 테스트, 작은 코드 조각 검증 |
| `works/02_experiments/` | 모델·데이터셋·하이퍼파라미터 비교 실험 |
| `works/03_projects/` | 정리된 tutorial, 재사용 가능한 예제, 문서화된 학습 단위 |
| `archive/` | 완료된 작업 폴더 보관 |

---

## 작업 폴더 규칙

`works/01_sandbox`, `works/02_experiments`, `works/03_projects` 아래의 작업 폴더는 동일한 네이밍 규칙과 내부 구조를 사용한다.

### 네이밍 규칙

```text
works/{category}/YYMMDD_{work-name-kebab-case}/
```

예시는 다음과 같다.

```text
works/01_sandbox/260603_torchvision-smoke-test/
works/02_experiments/260603_resnet-transfer-learning/
works/03_projects/260603_image-classification-tutorial/
```

### 공통 내부 구조

```text
YYMMDD_{work-name-kebab-case}/
├── README.md
├── ROADMAP.md
├── SPEC.md
├── sessions/
├── configs/
├── src/
├── notebooks/
├── scripts/
├── tests/
├── docs/
│   ├── myst.yml
│   └── contents/
└── outputs/
```

### 공통 내부 경로 역할

| 경로 | 역할 |
|------|------|
| `README.md` | 작업 개요, 실행 방법, 주요 결과 |
| `ROADMAP.md` | Stage-Phase-Task 진행 체크리스트 |
| `SPEC.md` | 목적, 범위, 데이터셋, 모델, 작업별 규칙, 평가 기준 |
| `sessions/` | 해당 작업의 세션 핸드오프 |
| `configs/` | YAML/JSON 설정 파일 |
| `src/` | Python source code |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | 실행용 shell/python scripts |
| `tests/` | pytest, smoke test |
| `docs/myst.yml` | 작업별 문서 목차 |
| `docs/contents/` | 작업별 문서 본문 |
| `outputs/` | logs, figures, checkpoints 등 산출물 |

### 작업 시작 전 확인

**MUST**
- 작업 폴더에서 작업을 시작하기 전 `SPEC.md`를 확인한다.
- `SPEC.md`에 작업별 규칙이 있으면 전역 지침에 추가하여 적용한다.
- 전역 지침과 충돌하면 `SPEC.md`의 작업별 규칙을 우선한다.

**MUST NOT**
- 작업 폴더 안에 별도 `rules/` 폴더를 생성하지 않는다.
- 작업별 규칙을 별도 파일로 분리하지 않는다. 필요한 규칙은 `SPEC.md`에 기록한다.

---

## 문서 관리 규칙

문서는 모두 `docs/contents/` 아래 저장하고, `docs/myst.yml`에서 목차를 정의한다.

### 전역 문서

| 항목 | 위치 |
|------|------|
| 문서 본문 | `_workspace/docs/contents/` |
| 문서 목차 | `_workspace/docs/myst.yml` |

### 작업별 문서

| 항목 | 위치 |
|------|------|
| 문서 본문 | `works/{category}/{work}/docs/contents/` |
| 문서 목차 | `works/{category}/{work}/docs/myst.yml` |

### 문서 추가·삭제 시

**MUST**
- `docs/contents/`에 문서를 추가하거나 삭제할 때 해당 `docs/myst.yml`의 `nav` 항목을 함께 업데이트한다.

목차 형식은 다음 기준을 따른다.

```yaml
site:
  nav:
    - title: 문서 제목
      file: contents/document-name
```

---

## 코드 작성 규칙

### Python

**MUST**
- PyTorch 중심으로 작성한다.
- tutorial 코드는 학습 가능성과 가독성을 우선한다.
- 실험 코드는 재현 가능성을 위해 seed, config, device 설정을 명시한다.
- 필요한 경우 type hint를 사용한다.
- 경로 처리는 `pathlib.Path`를 우선 사용한다.

**MUST NOT**
- 과도한 추상화로 tutorial 흐름을 숨기지 않는다.
- 대용량 파일 경로를 코드 내부에 무분별하게 하드코딩하지 않는다.

### 검증

코드 수정 후 가능한 경우 다음 중 하나 이상을 수행한다.

- import smoke test
- CUDA 사용 가능 여부 확인
- 단일 batch forward pass
- 짧은 epoch 실행
- pytest

---

## 산출물 및 Git 추적 정책

### 산출물

`outputs/`는 실행 결과를 저장하는 위치다.

예시는 다음과 같다.

```text
outputs/
├── logs/
├── figures/
└── checkpoints/
```

**MUST NOT**
- 대용량 checkpoint, dataset, pretrained backbone을 Git 추적 대상으로 추가하지 않는다.
- `outputs/`의 대용량 산출물을 사용자 승인 없이 커밋 대상으로 정리하지 않는다.

### Archive

완료된 작업 폴더는 원래 category를 유지한 채 `archive/` 아래로 이동한다.

```text
works/02_experiments/260603_resnet-transfer-learning/
archive/02_experiments/260603_resnet-transfer-learning/
```

이동 전 확인 기준은 다음과 같다.

- `ROADMAP.md`의 주요 Task가 완료되어 있다.
- `README.md`에 실행 방법과 주요 결과가 정리되어 있다.
- `SPEC.md`에 최종 범위와 평가 기준이 정리되어 있다.
- 필요한 문서가 `docs/contents/`에 있고 `docs/myst.yml` 목차가 갱신되어 있다.

---

## Git 정책

### 허용

읽기 전용 명령만 사용한다.

- `git status`
- `git log`
- `git diff`
- `git show`
- `git branch`

### 금지

**MUST NOT**
- `git commit`, `git push`, `git merge`, `git rebase`, `git reset`, `git checkout` 명령어를 실행하지 않는다.
- `git commit --amend`를 사용하지 않는다.
- `--no-verify` 옵션을 사용하지 않는다.
- 커밋·푸시·PR을 대신 수행하지 않는다.

### 커밋 메시지 형식

커밋 메시지는 사용자가 요청할 때 제안만 한다.

```text
{type}({scope}): {subject}
```

| 타입 | 용도 |
|------|------|
| `feat` | 새로운 기능·문서 추가 |
| `docs` | 기존 문서 수정·보완 |
| `fix` | 오류·오탈자 수정 |
| `refactor` | 구조 개선 |
| `chore` | 빌드·설정·유지보수 작업 |

---

## 명령어: session-handoff

사용자가 "세션 핸드오프", "session-handoff", "/session-handoff", "핸드오프 작성"이라고 요청하면 아래 절차를 실행한다.

### 저장 위치

- 작업 중인 `works/{category}/{work}/` 폴더가 명확한 경우: `works/{category}/{work}/sessions/YYMMDD-HHMMSS_session-handoff.md`
- 작업 중인 `works/{category}/{work}/` 폴더가 명확하지 않은 경우: `_workspace/sessions/YYMMDD-HHMMSS_session-handoff.md`

### 실행 절차

1. 저장 위치를 결정하고 사용자에게 먼저 알린다.
2. `sessions/` 폴더에 이전 handoff 파일이 있으면 파일명을 확인한다.
3. 현재 시각으로 파일명을 생성한다. 형식은 `YYMMDD-HHMMSS_session-handoff.md`이다.
4. 세션 전체를 분석하여 아래 구조의 문서를 작성한다.
5. 파일을 저장하고 경로를 사용자에게 알린다.

### 작성 원칙

- 단순 요약이 아니라 다른 세션에서 즉시 이어받을 수 있도록 작성한다.
- 확정된 내용과 미결 내용을 명확히 구분한다.
- 다음 작업은 바로 지시할 수 있는 수준으로 구체화한다.
- 기술 용어, 파일명, 경로는 정확하게 그대로 기록한다.

### 문서 구조

```text
# [주제] 세션 핸드오프

> 작성일: YYYY-MM-DD
> 세션 목적: (한 줄)
> 이전 핸드오프: {파일명, 없으면 이 줄 생략}

---

## 1. 세션 핵심 요약

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---------|---------|

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|------|---------|------|

## 4. AI 핵심 제안 및 분석

## 5. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|------|---------|--------------|

## 6. 다음 작업 목록

| 우선순위 | 작업 | 관련 폴더/파일 | 사용할 도구 |
|---------|------|--------------|----------|

## 7. 참고 자료 및 링크

| 항목 | 위치/경로 | 용도 |
|------|---------|------|
```

내용이 없는 섹션은 생략한다.

---

## 명령어: commit-message

사용자가 "커밋 메시지", "commit-message", "/commit-message"라고 요청하면 아래 절차를 실행한다.

커밋은 사용자가 직접 수행한다. Codex는 메시지만 제안한다.

1. `git status`로 변경 파일 목록을 확인한다.
2. `git diff HEAD --stat`로 변경 규모를 파악한다.
3. 변경 규모가 크면 주요 파일만 `git diff HEAD -- {파일}`로 선택 확인한다.
4. 커밋 메시지를 제안한다. 변경 규모가 클 경우 단일·분리 두 가지를 함께 제안한다.

형식은 다음과 같다.

```text
{type}({scope}): {subject}
```
