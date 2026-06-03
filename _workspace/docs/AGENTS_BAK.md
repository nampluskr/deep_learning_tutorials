# techbase — Codex 지침

이 워크스페이스는 techbase의 기술 학습·연구 자료를 작성·관리한다.
PARA 메서드 기반 문서 관리 체계로 운영한다.

## AI 역할 범위

- 파일 작성·수정 보조 역할에 한정한다.
- 워크스페이스 구조·원칙 변경은 사용자 결정에 따른다.
- 사용자 지시 없이 파일을 생성하거나 구조를 변경하지 않는다.

---

## 협업 규칙

### 파일 생성·수정 절차

**MUST**
- 파일 생성·수정 전 경로와 내용을 먼저 제안한다.
- 사용자 승인 후 실행한다.
- 여러 파일을 수정할 경우 수정 범위를 먼저 설명하고 진행 여부를 확인한다.

**MUST NOT**
- 사용자의 명시적 승인 없이 파일·폴더를 생성하지 않는다.
- "검토해 주세요"는 생성 승인이 아니다.
- 구조 변경(폴더 생성, 파일 이동, 삭제)은 반드시 사용자 확인 후 실행한다.

### 응답 스타일

**MUST NOT**
- 이모지를 사용하지 않는다.
- 불필요한 칭찬·감탄 표현을 사용하지 않는다.

**MUST**
- 간결하게 응답한다.
- WHY가 명확하지 않으면 코드 주석을 작성하지 않는다.

### 편집 환경 표준

- 인코딩: UTF-8
- 줄 끝: LF
- 들여쓰기: 2-space

---

## Git 정책

### 허용 (읽기 전용)

- `git status`, `git log`, `git diff`, `git show`, `git branch`

### MUST NOT

- `git commit`, `git push`, `git merge`, `git rebase`, `git reset`, `git checkout` 명령어를 실행하지 않는다.
- 커밋·푸시·PR은 사용자가 직접 수행한다.
- `git commit --amend`를 사용하지 않는다.
- `--no-verify` 옵션을 사용하지 않는다.

### 커밋 메시지 형식

`{type}({scope}): {subject}`

| 타입 | 용도 |
|------|------|
| `feat` | 새로운 기능·문서 추가 |
| `docs` | 기존 문서 수정·보완 |
| `fix` | 오류·오탈자 수정 |
| `refactor` | 구조 개선 (기능 변경 없음) |
| `chore` | 빌드·설정·유지보수 작업 |

---

## 네이밍 컨벤션

| 폴더 | 형식 | 예시 |
|------|------|------|
| `100_inbox/01_notes/` | `YYMMDD_키워드.md` | `260530_linear-algebra.md` |
| `100_inbox/02_refs/` | `YYMMDD_키워드.md` | `260530_pytorch-guide.md` |
| `100_inbox/03_courses/` | `YYMMDD_{강좌명-kebab-case}/` | `260530_fast-ai-course/` |
| `200_daily/01_tasks/` | `YYMMDD-tasks.md` | `260530-tasks.md` |
| `500_works/01_sandbox/` | `YYMMDD_{키워드-kebab-case}/` | `260530_mnist-test/` |
| `500_works/02_experiments/` | `YYMMDD_{키워드-kebab-case}/` | `260530_image-classification-eval/` |
| `500_works/03_projects/` | `YYMMDD_{프로젝트명-kebab-case}/` | `260530_linear-algebra-notes/` |
| `300_resources/`, `400_areas/`, `600_legacy/`, `900_archive/` | `키워드-키워드.md` | `matrix-decomposition.md` |

### Subject 폴더 체계

`300_resources/`, `400_areas/`, `600_legacy/`, `900_archive/` 하위 주제 폴더는 `NN_{코드}_{이름}` 형식으로 명명한다.

| 번호 | 코드 | 분야 |
|------|------|------|
| 01 | MATH | Mathematics |
| 02 | PHYS | Physics |
| 03 | NA   | Numerical Analysis |
| 04 | ML   | Machine Learning |
| 05 | CV   | Computer Vision |
| 06 | NLP  | Natural Language Processing |
| 07 | CS   | Computer Science |
| 08 | DEV  | Development Environment and Tools |
| 09 | MISC | Miscellaneous |

---

## 문서 작성 스타일

### Frontmatter 표준

```yaml
---
title: ""
date: "YYYY-MM-DD"
type: ""          # note / project / resource / area / daily
topic: ""
tags: []
status: ""        # draft / review / done
execute: false
math: false
has_images: false
source: ""
---
```

### 언어

- 기본 언어: 한국어
- 기술 용어·고유명사·파일명·경로는 영어 원문 그대로 사용한다.
- 한국어 번역이 부자연스러운 용어는 영어를 유지한다.

### 문장 스타일

- 종결 표현: `~한다` 체 (명사형 종결 금지)
- 이모지를 사용하지 않는다.

### 제목 계층

- H1 (`#`): 문서 제목 — 파일당 하나
- H2 (`##`): 주요 섹션
- H3 (`###`): 섹션 내 세부 항목
- H4 이하는 사용하지 않는다.

### 목록 vs 표

| 상황 | 형식 |
|------|------|
| 항목 간 관계·비교가 필요한 경우 | 표 |
| 단순 나열 | 불릿 목록 |
| 순서가 있는 절차 | 번호 목록 |

### 코드블록

- 파일 경로, 명령어, 코드는 반드시 코드블록 또는 인라인 코드로 표기한다.
- 코드블록에는 언어를 명시한다.

### 강조

- 굵게 (`**텍스트**`): 핵심 용어, 중요 경고
- 기울임: 사용하지 않는다.

---

## 프로젝트 규칙

`500_works/03_projects/` 하위 프로젝트는 개별 `rules/` 폴더를 가질 수 있다.

**MUST**
- 프로젝트 폴더 내 작업 시작 전 `rules/project-rules.md`가 있는지 확인한다.
- 존재하면 전역 규칙에 추가하여 적용한다.
- 충돌 시 프로젝트 규칙이 전역 규칙보다 우선한다.

**MUST NOT**
- 프로젝트 규칙에 없는 작업 범위를 임의로 확장하지 않는다.

### 프로젝트 폴더 구조

```
YYMMDD_{프로젝트명}/
├── README.md                  (프로젝트 포털 — 개요·링크)
├── ROADMAP.md                 (진행 현황 — Stage-Phase-Task 체크박스)
├── SPEC.md                    (문서 명세서 — 배경·목차·참고자료)
├── sessions/                  (세션 핸드오프 — YYMMDD-HHMMSS_session-handoff.md)
├── rules/                     (프로젝트 전용 규칙 — 선택사항)
│   └── project-rules.md
└── docs/
    ├── myst.yml               (Jupyter Book v2 설정)
    └── contents/              (문서 파일)
        └── {kebab-case-name}_v{x.x}.md
```

### 프로젝트 로드맵 계층 구조

```
Stage   →  큰 작업 단위 (설계 / 구현 / 검증 등)
Phase   →  Stage 내 세부 단계
Task    →  Phase 내 개별 체크박스 항목 (x.x 번호)
```

### `docs/contents/` 파일 추가 시

`docs/contents/`에 파일을 추가하거나 삭제할 때 반드시 `docs/myst.yml`의 `nav` 항목을 함께 업데이트한다.

```yaml
site:
  nav:
    - title: {문서 제목}
      file: contents/{kebab-case-name}_v{x.x}
```

---

## 명령어: session-handoff

사용자가 "세션 핸드오프", "session-handoff", "핸드오프 작성"이라고 요청하면 아래 절차를 실행한다.

### 저장 위치

- 작업 중인 프로젝트가 명확한 경우: `500_works/03_projects/{project}/sessions/YYMMDD-HHMMSS_session-handoff.md`
- 프로젝트 무관 세션: `_workspace/sessions/YYMMDD-HHMMSS_session-handoff.md`

### 실행 절차

1. 저장 위치를 결정하고 사용자에게 먼저 알린다.
2. `sessions/` 폴더에 이전 handoff 파일이 있으면 파일명을 확인한다.
3. 현재 시각으로 파일명을 생성한다. 형식: `YYMMDD-HHMMSS_session-handoff.md`
4. 세션 전체를 분석하여 아래 구조의 문서를 작성한다.
5. 파일을 저장하고 경로를 사용자에게 알린다.

### 작성 원칙

- 단순 요약이 아니라 다른 세션에서 즉시 이어받을 수 있도록 작성한다.
- 확정된 내용과 미결 내용을 명확히 구분한다.
- 다음 작업은 바로 지시할 수 있는 수준으로 구체화한다.
- 기술 용어, 파일명, 경로는 정확하게 그대로 기록한다.

### 문서 구조

```
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

사용자가 "커밋 메시지", "commit-message"라고 요청하면 아래 절차를 실행한다.

커밋은 사용자가 직접 수행한다. 메시지만 제안한다.

1. `git status`로 변경 파일 목록을 확인한다.
2. `git diff HEAD --stat`로 변경 규모를 파악한다.
3. 변경 규모가 크면 주요 파일만 `git diff HEAD -- {파일}`로 선택 확인한다.
4. 커밋 메시지를 제안한다. 변경 규모가 클 경우 단일·분리 두 가지를 함께 제안한다.

형식: `{type}({scope}): {subject}` + 복수 변경 시 bullet 요약