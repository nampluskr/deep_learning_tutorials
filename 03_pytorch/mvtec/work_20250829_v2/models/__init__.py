# Base components
from .model_base import (
    TimmFeatureExtractor,
    dryrun_find_featuremap_dims,
    get_local_weight_path,
    load_backbone_weights,
    get_feature_extractor,
    BACKBONE_WEIGHT_FILES
)

# Autoencoder models
from .model_ae import (
    VanillaAE,
    UNetAE,
    AELoss,
    AECombinedLoss,
    compute_ae_anomaly_map,
    compute_ae_anomaly_scores,
    get_ae_model,
    get_ae_loss
)

# STFPM models (when implemented)
# from .model_stfpm import (
#     STFPMModel,
#     STFPMLoss,
#     AnomalyMapGenerator,
#     InferenceBatch
# )

# Future models (planned)
# from .model_fastflow import *
# from .model_padim import *
# from .model_patchcore import *
# from .model_vae import *

__all__ = [
    # Base components
    "TimmFeatureExtractor",
    "dryrun_find_featuremap_dims", 
    "get_local_weight_path",
    "load_backbone_weights",
    "get_feature_extractor",
    "BACKBONE_WEIGHT_FILES",
    
    # Autoencoder models
    "VanillaAE",
    "UNetAE", 
    "AELoss",
    "AECombinedLoss",
    "compute_ae_anomaly_map",
    "compute_ae_anomaly_scores",
    "get_ae_model",
    "get_ae_loss",
    
    # STFPM models (when implemented)
    # "STFPMModel",
    # "STFPMLoss", 
    # "AnomalyMapGenerator",
    # "InferenceBatch",
]