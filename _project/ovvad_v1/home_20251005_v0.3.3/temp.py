import timm
import torch
from pathlib import Path
from safetensors.torch import load_file

# def load_model_from_local_hf_cache(model_name, backbone_dir, cache_subdir):
#     import os
#     import timm
#     from safetensors.torch import load_file

#     model = timm.create_model(model_name, pretrained=False)
#     cache_dir = os.path.join(backbone_dir, cache_subdir, "snapshots")
#     if not os.path.isdir(cache_dir):
#         raise FileNotFoundError(f"Snapshots directory not found: {cache_dir}")

#     snapshot_dirs = os.listdir(cache_dir)
#     if not snapshot_dirs:
#         raise FileNotFoundError(f"No snapshots found in {cache_dir}")

#     snapshot_hash = snapshot_dirs[0]
#     weight_path = os.path.join(cache_dir, snapshot_hash, "model.safetensors")
#     if not os.path.isfile(weight_path):
#         raise FileNotFoundError(f"Weight file not found: {weight_path}")

#     print(f"Loading {model_name} from: {weight_path}")
#     state_dict = load_file(weight_path)
#     model.load_state_dict(state_dict)
#     return model

# # 사용
# backbone_dir = "/mnt/d/backbones"

# model_cait = load_model_from_local_hf_cache(
#     'cait_m48_448',
#     backbone_dir,
#     'cait_m48_448.fb_dist_in1k'
# )

# model_deit = load_model_from_local_hf_cache(
#     'deit_base_distilled_patch16_384',
#     backbone_dir,
#     'deit_base_distilled_patch16_384.fb_in1k'
# )

# 1. 모델 구조 생성 (pretrained=False)
model_cait = timm.create_model('cait_s24_224', pretrained=True)
# model_cait = timm.create_model('cait_m48_448', pretrained=False)
# model_deit = timm.create_model('deit_base_distilled_patch16_384', pretrained=False)

# # 2. backbones 디렉토리 설정
# backbone_dir = Path("/mnt/d/backbones")

# # 3. CaiT 모델 weights 로드
# cait_cache_dir = backbone_dir / "cait_m48_448.fb_dist_in1k"
# cait_snapshots = cait_cache_dir / "snapshots"
# cait_snapshot_hash = list(cait_snapshots.iterdir())[0]  # 첫 번째 해시 디렉토리
# cait_weight_path = cait_snapshot_hash / "model.safetensors"

# if cait_weight_path.exists():
#     print(f"Loading CaiT from: {cait_weight_path}")
#     cait_state_dict = load_file(cait_weight_path)
#     model_cait.load_state_dict(cait_state_dict)
# else:
#     print(f"CaiT weight file not found: {cait_weight_path}")

# # 4. DeiT 모델 weights 로드
# deit_cache_dir = backbone_dir / "deit_base_distilled_patch16_384.fb_in1k"
# deit_snapshots = deit_cache_dir / "snapshots"
# deit_snapshot_hash = list(deit_snapshots.iterdir())[0]  # 첫 번째 해시 디렉토리
# deit_weight_path = deit_snapshot_hash / "model.safetensors"

# if deit_weight_path.exists():
#     print(f"Loading DeiT from: {deit_weight_path}")
#     deit_state_dict = load_file(deit_weight_path)
#     model_deit.load_state_dict(deit_state_dict)
# else:
#     print(f"DeiT weight file not found: {deit_weight_path}")

# # 5. 모델을 eval 모드로 전환
# model_cait.eval()
# model_deit.eval()