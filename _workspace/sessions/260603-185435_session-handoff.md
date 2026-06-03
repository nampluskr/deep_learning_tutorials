# 워크스페이스 지침 재설정 세션 핸드오프

> 작성일: 2026-06-03
> 세션 목적: `deep_learning_tutorials` 워크스페이스를 PARA 문서 관리용 지침에서 PyTorch + CUDA 기반 Computer Vision tutorial 작업 지침으로 재설정한다.

---

## 1. 세션 핵심 요약

현재 워크스페이스의 기존 `AGENTS.md`는 PARA 방식의 Markdown 문서 관리 지침이었다.
사용자는 이 워크스페이스를 WSL + Anaconda + PyTorch + CUDA + VS Code + Codex 기반 Computer Vision tutorial 예제 테스트 공간으로 운영하려고 한다.

이에 따라 기존 `AGENTS.md`를 `_workspace/docs/AGENTS_BAK.md`로 이동해 백업하고, 새 `AGENTS.md`에 딥러닝 tutorial 작업 구조와 운영 지침을 반영했다.
루트에는 `_workspace/` 폴더를 만들고 하위에 `docs/`, `sessions/`를 생성했다.

## 2. 사용자 요청 및 의도

| 요청 내용 | 배경 목적 |
|---------|---------|
| 기존 PARA 지침 검토 | 현재 목적과 기존 문서 관리 지침의 불일치 확인 |
| tutorial 작업은 `works/` 아래 `01_sandbox`, `02_experiments`, `03_projects`로 구분 | 작업 성격별 관리 체계 수립 |
| 루트 `tutorials/`, 루트 `src/` 제거 | 모든 작업 단위를 개별 작업 폴더 내부에서 관리 |
| 루트 `_workspace/docs`, `_workspace/sessions` 생성 | 글로벌 참고 문서와 작업 무관 session handoff 관리 |
| 모든 `works` 작업 폴더 구조 통일 | sandbox, experiments, projects 간 동일한 운영 기준 적용 |
| 작업별 `rules/` 제거, 규칙은 `SPEC.md`에 기록 | 작업별 규칙 파일을 단순화하고 명세 문서로 통합 |
| 문서는 `docs/contents/`, 목차는 `docs/myst.yml`에 관리 | MyST 기반 문서 목차 관리 기준 유지 |
| 루트 `archive/` 도입 | 완료된 작업 폴더를 원래 category 유지 상태로 보관 |
| CUDA 공통 사용 및 `pytorch_env` 명시 | 실행 환경을 Anaconda PyTorch + CUDA 기준으로 고정 |
| `/session-handoff`, `/commit-message` 명령 반영 | slash command 형태의 요청도 인식하도록 지침 보완 |

## 3. 확정된 결정사항

| 항목 | 확정 내용 | 비고 |
|------|---------|------|
| 워크스페이스 목적 | PyTorch + CUDA 기반 Computer Vision tutorial 예제 작성·실행·검증 | 기존 PARA 문서 관리 목적에서 변경 |
| 기본 conda 환경 | `pytorch_env` | 실행 전 `conda activate pytorch_env` 전제 |
| CUDA | 공통 필수 전제 | CPU fallback은 기본 제공하지 않음 |
| 외부 dataset 경로 | `DATASET_DIR` | 워크스페이스 외부 `datasets/` 참조 |
| 외부 pretrained backbone 경로 | `BACKBONE_DIR` | 워크스페이스 외부 `backbones/` 참조 |
| 루트 관리 폴더 | `_workspace/docs`, `_workspace/sessions` | 글로벌 문서와 작업 무관 handoff 관리 |
| 작업 폴더 위치 | `works/01_sandbox`, `works/02_experiments`, `works/03_projects` | 작업 성격별 구분 |
| 작업 폴더 네이밍 | `YYMMDD_{work-name-kebab-case}` | 세 category에 동일 적용 |
| 작업 폴더 내부 구조 | `README.md`, `ROADMAP.md`, `SPEC.md`, `sessions/`, `configs/`, `src/`, `notebooks/`, `scripts/`, `tests/`, `docs/`, `outputs/` | 모든 작업 폴더에 동일 적용 |
| 작업별 규칙 | `SPEC.md`에 기록 | 별도 `rules/` 폴더 없음 |
| 문서 저장 위치 | `docs/contents/` | 전역과 작업별 모두 동일 |
| 문서 목차 | `docs/myst.yml` | 문서 추가·삭제 시 함께 갱신 |
| 완료 작업 보관 | `archive/{category}/` | 원래 category 유지 |
| Git 정책 | 읽기 전용 명령만 사용 | commit/push/merge/rebase/reset/checkout 금지 |
| session-handoff 명령 | `세션 핸드오프`, `session-handoff`, `/session-handoff`, `핸드오프 작성` | 작업 폴더가 불명확하면 `_workspace/sessions/` 저장 |
| commit-message 명령 | `커밋 메시지`, `commit-message`, `/commit-message` | 메시지만 제안 |

## 4. AI 핵심 제안 및 분석

- 기존 지침은 PARA 기반 지식 관리에 적합하지만, 현재 저장소 목적에는 맞지 않는다고 분석했다.
- 루트 `tutorials/`와 루트 `src/`를 두지 않고, 모든 코드와 문서를 개별 작업 폴더 안에서 관리하는 구조가 확정됐다.
- `works/01_sandbox`, `works/02_experiments`, `works/03_projects`는 작업 성격만 다르고 내부 구조와 운영 기준은 동일하게 적용한다.
- `_workspace/`는 개별 작업 폴더를 관리하기 위한 글로벌 영역으로 정의됐다.
- `_workspace/docs/AGENTS_BAK.md`에는 기존 PARA 지침을 백업해 전역 참고 문서로 보관했다.
- `AGENTS.md`에는 CUDA 필수 실행 기준과 `pytorch_env` conda 환경 기준을 명확히 반영했다.

## 5. 미결 사항

| # | 항목 | 현재 상태 | 결정 필요 내용 |
|---|------|---------|--------------|
| 1 | 실제 루트 폴더 구조 생성 | `_workspace/docs`, `_workspace/sessions`만 생성됨 | `works/`, `archive/`, `.vscode/` 등 나머지 폴더 생성 여부 |
| 2 | `README.md` 개정 | 기존 단순 README 유지 | 새 워크스페이스 목적과 구조 반영 여부 |
| 3 | `environment.yml` 생성 | 아직 없음 | `pytorch_env` 기준 의존성 목록 확정 |
| 4 | `.gitignore` 생성 | 아직 없음 | dataset, backbone, outputs, checkpoints 등 제외 규칙 확정 |
| 5 | `_workspace/docs/myst.yml`와 `contents/` 생성 | 아직 없음 | 전역 문서 목차 관리 파일 생성 여부 |
| 6 | 작업 폴더 template | 아직 없음 | `works` 하위 작업 폴더 생성용 template 운영 여부 |

## 6. 다음 작업 목록

| 우선순위 | 작업 | 관련 폴더/파일 | 사용할 도구 |
|---------|------|--------------|----------|
| 1 | `README.md`를 새 워크스페이스 목적 기준으로 개정 | `README.md` | 파일 수정 |
| 2 | 루트 기본 폴더 생성 여부 결정 | `works/`, `archive/`, `.vscode/` | 사용자 승인 후 폴더 생성 |
| 3 | `environment.yml` 초안 작성 | `environment.yml` | 사용자 승인 후 파일 생성 |
| 4 | `.gitignore` 초안 작성 | `.gitignore` | 사용자 승인 후 파일 생성 |
| 5 | `_workspace/docs` 문서 체계 초기화 | `_workspace/docs/myst.yml`, `_workspace/docs/contents/` | 사용자 승인 후 파일·폴더 생성 |
| 6 | 첫 tutorial 작업 폴더 생성 기준 확정 | `works/{category}/YYMMDD_{work-name-kebab-case}/` | 사용자 승인 후 폴더 생성 |

## 7. 참고 자료 및 링크

| 항목 | 위치/경로 | 용도 |
|------|---------|------|
| 현재 전역 지침 | `AGENTS.md` | Codex 작업 규칙 |
| 기존 PARA 지침 백업 | `_workspace/docs/AGENTS_BAK.md` | 이전 지침 참고 |
| 글로벌 세션 폴더 | `_workspace/sessions/` | 작업 무관 session handoff 저장 |
