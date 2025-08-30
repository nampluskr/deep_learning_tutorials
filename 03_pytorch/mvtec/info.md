
##### 프로젝트 개요
- OLED 디스플레이의 얼룩(stain), 무라(mura) 및 화질 이상(display quality defects) 검출
- 사람이 인지가능한 미세한 시각적 차이를 가진 anomaly 검출
- OLED 디스플레이의 얼룩 검출은 전형적인 Anomaly Detection 문제
- 정상 분포의 미세한 일탈에 강해야 함
- 정상 데이터만으로 학습해야 하는 One-Class 학습 상황
- Vanilla Simple Autoencoder부터 시작해서, SOTA Vision Anomaly Detection 모델들과 단계적 평가
- 패턴에 의존하지 않는 다양한 콘텐츠와 휘도 조건에서의 OLED 화질 이상 검출
- 오픈 데이터셋을 활용하여 모델 아키텍쳐와 성능 개선 평가를 진행하고, 실제 데이터 적용 평가


##### 데이터 특성
- 정상 데이터가 압도적으로 많고 불량이 희소하며 모양·위치·수준이 예측 불가능
- OLED 디스플레이에 따라 해상도 (height, width)가 다름
- 정상 데이터 중심의 불균형 데이터 환경 대응
- 극도로 불균형한 데이터 분포 (정상 >> 불량)
- 불량 데이터의 형태, 위치, 수준(강도) 예측 불가
- 구조적, 저주파, 고주파 무라를 포함
- 측정 데이터는 특정 폴더에 저장되어 있고 실시간 처리 이슈 없음

#### 추가 정보
- 사용자는 Computer Vision 연구자로 PyTorch를 사용
- 기술적으로 깊이 있는 설명을 필요함

#### 모델 개발 단계

1. Baseline 모델 구축 - Vanilla CNN Autoencoder
  - Encoder-Decoder 아키텍처 설계
  - Reconstruction loss 기반 anomaly detection
  - 하이퍼파라미터 최적화

Vanila CNN Autoencoder 를 사용하여, MVTec 데이터셋의 카테고리별 Anomaly Detection 성능을 도출하고, 시각화하는 프로세스를 단계별 상세 psedo 코드 형태로 작성해 주세요.
(실제 파이썬 코드는 추가 요청)

[예시]
- 이미지 데이터 Augmentation 정의: train / test transform
- MVTec Dataset 정의: train / valid / test dataset
- dataloader 정의: train / valid / test dataloader
- Vanila Autoencoder 정의: encoder / decoder
- loss / metric 함수 정의
- 정상 데이터 만으로 학습: training + validation + early stopping + learning rate 스케쥴링
- 테스트 데이터를 대상으로 카테고리별 성능 평가
- 특정 케테고리별 정상/불량 이미지 예시 시각화

필요한 부분만 수정/구현하고, 추가적인 함수/클래스 생성은 하지 말아주세요. 요청한 내용만 작성해 주세요.

### OLED 화질이상의 특징

- 미세한 휘도/색 불균일 (무라, mura)
- 얼룩, 변색 (blotch, stain)
- 저주파 패턴 불균일 (cloud mura, banding)
- 표면 질감 변화 (texture variation)
- 미세한 색차 (color deviation)

#### MVTec AD 데이터셋 추천 카테고리
높은 유사성

- carpet - 직물 표면의 색상 불균일, 얼룩, 질감 변화
- leather - 가죽 표면의 색상 불균일, 얼룩, 표면 질감
- tile - 타일 표면의 색상 변화, 얼룩, 균일도 문제
- wood - 나무 표면의 색상 불균일, 결 변화

중간 유사성

- fabric - 직물의 색상/질감 변화, 패턴 불균일
- grid - 격자 패턴의 불균일, 주기적 구조 변화
- bottle - 투명/반투명 표면의 얼룩, 변색

#### VisA 데이터셋 추천 카테고리
높은 유사성

- PCB1, PCB2, PCB3, PCB4 - 회로기판 표면의 색상 불균일, 얼룩
- macaroni1, macaroni2 - 표면 색상 변화, 질감 불균일
- capsules - 캡슐 표면의 색상 변화, 얼룩

중간 유사성

- candle - 표면 색상 불균일, 왁스 얼룩
- cashew - 견과류 표면 색상 변화
- pipe_fryum - 파이프 표면 불균일

#### BTAD 데이터셋 추천 카테고리
높은 유사성

- 01 - 플라스틱/금속 표면 결함 (일반적으로 표면 불균일)
- 02 - 산업용 부품 표면 변색, 얼룩
- 03 - 제품 표면의 질감/색상 변화