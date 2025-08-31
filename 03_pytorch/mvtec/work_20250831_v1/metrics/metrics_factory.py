import torch
from typing import Dict, List, Union, Optional

# Model type mappings for auto-detection
MEMORY_BASED_MODELS = {
    'padim', 'patchcore', 'spade', 'cfa', 'patchcore_plus'
}

GRADIENT_BASED_MODELS = {
    'ae', 'autoencoder', 'vanilla_ae', 'unet_ae',
    'stfpm', 'draem', 'dfm', 'ganomaly', 'dfkde', 'reverse_distillation'
}

FLOW_BASED_MODELS = {
    'fastflow', 'normalizing_flow', 'cflow', 'realvnf'
}

OLED_MODELS = {
    'oled_ae', 'oled_padim', 'oled_stfpm', 'oled_custom'
}

# Model-specific metric configurations
MODEL_TYPE_METRICS = {
    'memory_based': {
        'core': ['auroc', 'aupr', 'f1', 'threshold'],
        'specialized': ['mahalanobis_dist', 'memory_efficiency'],
        'optional': ['accuracy', 'precision', 'recall']
    },
    'gradient_based': {
        'core': ['auroc', 'aupr', 'psnr', 'ssim'],
        'specialized': ['lpips', 'feature_similarity'],
        'optional': ['f1', 'threshold', 'perceptual_loss']
    },
    'flow_based': {
        'core': ['auroc', 'aupr', 'log_likelihood'],
        'specialized': ['bpd', 'flow_jacobian'],
        'optional': ['f1', 'threshold']
    },
    'oled': {
        'core': ['auroc', 'aupr', 'delta_e2000'],
        'specialized': ['fft_mura', 'connected_component', 'luminance_uniformity'],
        'optional': ['psnr', 'ssim', 'color_consistency']
    }
}


def detect_model_type(model_name: str) -> str:
    """Auto-detect model type from model name."""
    model_name = model_name.lower()
    
    # Check OLED models first (more specific)
    if any(oled_model in model_name for oled_model in OLED_MODELS):
        return 'oled'
    
    if any(memory_model in model_name for memory_model in MEMORY_BASED_MODELS):
        return 'memory_based'
    elif any(gradient_model in model_name for gradient_model in GRADIENT_BASED_MODELS):
        return 'gradient_based'  
    elif any(flow_model in model_name for flow_model in FLOW_BASED_MODELS):
        return 'flow_based'
    else:
        # Default to gradient_based for unknown models
        return 'gradient_based'


def get_metric(name: str, **params):
    """Factory function to create individual metric instances."""
    from .metrics_base import (
        AUROCMetric, AUPRMetric, AccuracyMetric, PrecisionMetric, 
        RecallMetric, F1Metric, OptimalThresholdMetric
    )
    from .metrics_memory import MahalanobisDistanceMetric, MemoryEfficiencyMetric
    from .metrics_gradient import PSNRMetric, SSIMMetric, LPIPSMetric, FeatureSimilarityMetric
    from .metrics_flow import LogLikelihoodMetric, BPDMetric, FlowJacobianMetric
    from .metrics_oled import (
        DeltaE2000Metric, FFTMuraMetric, ConnectedComponentMetric, 
        LuminanceUniformityMetric, ColorConsistencyMetric
    )
    
    # Metric registry
    METRIC_REGISTRY = {
        # Base metrics
        'auroc': AUROCMetric,
        'auc': AUROCMetric,
        'aupr': AUPRMetric, 
        'ap': AUPRMetric,
        'accuracy': AccuracyMetric,
        'precision': PrecisionMetric,
        'recall': RecallMetric,
        'f1': F1Metric,
        'threshold': OptimalThresholdMetric,
        
        # Memory-based metrics
        'mahalanobis_dist': MahalanobisDistanceMetric,
        'memory_efficiency': MemoryEfficiencyMetric,
        
        # Gradient-based metrics  
        'psnr': PSNRMetric,
        'ssim': SSIMMetric,
        'lpips': LPIPSMetric,
        'feature_similarity': FeatureSimilarityMetric,
        
        # Flow-based metrics
        'log_likelihood': LogLikelihoodMetric,
        'bpd': BPDMetric,
        'flow_jacobian': FlowJacobianMetric,
        
        # OLED-specific metrics
        'delta_e2000': DeltaE2000Metric,
        'fft_mura': FFTMuraMetric,
        'connected_component': ConnectedComponentMetric,
        'luminance_uniformity': LuminanceUniformityMetric,
        'color_consistency': ColorConsistencyMetric,
    }
    
    name = name.lower()
    if name not in METRIC_REGISTRY:
        available_names = list(METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric: {name}. Available metrics: {available_names}")
    
    metric_class = METRIC_REGISTRY[name]
    return metric_class(**params)


def get_metrics_for_model(
    model_name: str, 
    model_type: Optional[str] = None,
    include_optional: bool = False,
    custom_metrics: Optional[List[str]] = None
) -> Dict[str, torch.nn.Module]:
    """Get appropriate metrics for a specific model."""
    
    # Auto-detect model type if not provided
    if model_type is None:
        model_type = detect_model_type(model_name)
    
    if model_type not in MODEL_TYPE_METRICS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_TYPE_METRICS.keys())}")
    
    config = MODEL_TYPE_METRICS[model_type]
    
    # Start with core metrics
    metric_names = config['core'].copy()
    
    # Add specialized metrics
    metric_names.extend(config['specialized'])
    
    # Add optional metrics if requested
    if include_optional:
        metric_names.extend(config['optional'])
    
    # Add custom metrics if provided
    if custom_metrics:
        metric_names.extend(custom_metrics)
    
    # Remove duplicates while preserving order
    metric_names = list(dict.fromkeys(metric_names))
    
    # Create metric instances
    metrics = {}
    for metric_name in metric_names:
        try:
            metrics[metric_name] = get_metric(metric_name)
        except ValueError as e:
            print(f"Warning: Could not create metric '{metric_name}': {e}")
            continue
    
    return metrics


def get_default_metrics_config(model_type: str) -> Dict:
    """Get default metrics configuration for model type."""
    if model_type not in MODEL_TYPE_METRICS:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_TYPE_METRICS[model_type].copy()


# Example usage functions
def create_memory_based_metrics(**params):
    """Create standard metrics for memory-based models."""
    return get_metrics_for_model("padim", model_type="memory_based", **params)

def create_gradient_based_metrics(**params):  
    """Create standard metrics for gradient-based models."""
    return get_metrics_for_model("ae", model_type="gradient_based", **params)

def create_flow_based_metrics(**params):
    """Create standard metrics for flow-based models."""
    return get_metrics_for_model("fastflow", model_type="flow_based", **params)

def create_oled_metrics(**params):
    """Create standard metrics for OLED-specific models.""" 
    return get_metrics_for_model("oled_ae", model_type="oled", **params)