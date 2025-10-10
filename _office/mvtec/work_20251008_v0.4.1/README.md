### File Update
- dataloader.py
- registery.py
- main.py

### Model Update
- Dinomaly: 신규 생성
  - self.global_step 확인
  - model weights (regular): 추가 완료
    - dinov2_vit_small_14: https://huggingface.co/FoundationVision/unitok_external/blob/main/dinov2_vits14_pretrain.pth
    - dinov2_vit_base_14:  https://huggingface.co/spaces/BoukamchaSmartVisions/FeatureMatching/blob/main/models/dinov2_vitb14_pretrain.pth
    - dinov2_vit_large_14: https://huggingface.co/Cusyoung/CrossEarth/blob/main/dinov2_vitl14_pretrain.pth
  - model weights (reg): 테스트 할 것
    - dinov2reg_vit_small_14: https://dl.fbaipublicfiles.com/dinov2/dinov2_vit_s_14/dinov2_vit_s_14_reg4_pretrain.pth
    - dinov2reg_vit_base_14:  https://dl.fbaipublicfiles.com/dinov2/dinov2_vit_b_14/dinov2_vit_b_14_reg4_pretrain.pth
    - dinov2reg_vit_large_14: https://dl.fbaipublicfiles.com/dinov2/dinov2_vit_l_14/dinov2_vit_l_14_reg4_pretrain.pth
- fastflow: cait_m48_448 / deit_base_distilled_patch16_384 허깅페이스에서 다운로드 해서 테스트
  - https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k/tree/main
  - https://huggingface.co/timm/cait_m48_448.fb_dist_in1k/tree/main
- uflow: cait_m48_448 / cait_s24_224 허깅페이스에서 다운로드 해서 테스트
  - 스케쥴러 업데이트 출력 조정
