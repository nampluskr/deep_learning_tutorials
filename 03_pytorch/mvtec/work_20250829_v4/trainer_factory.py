from trainer_base import BaseTrainer, get_scheduler, get_stopper
from trainer_memory import MemoryTrainer
from trainer_gradient import GradientTrainer
from trainer_flow import FlowTrainer


# Model type mappings for auto-detection
MEMORY_BASED_MODELS = {
    'padim', 'patchcore', 'spade', 'cfa'
}

GRADIENT_BASED_MODELS = {
    'ae', 'autoencoder', 'vanilla_ae', 'unet_ae',
    'stfpm', 'draem', 'dfm', 'ganomaly', 'dfkde'
}

FLOW_BASED_MODELS = {
    'fastflow', 'normalizing_flow', 'cflow'
}


def infer_trainer_type_from_modeler(modeler):
    """Infer trainer type from modeler class and attributes (based on existing code)"""
    modeler_class_name = modeler.__class__.__name__.lower()
    
    # Check by modeler class name (primary method)
    if 'padim' in modeler_class_name or 'patchcore' in modeler_class_name or 'spade' in modeler_class_name:
        return 'memory'
    elif 'ae' in modeler_class_name or 'stfpm' in modeler_class_name or 'draem' in modeler_class_name:
        return 'gradient'
    elif 'flow' in modeler_class_name:
        return 'flow'
    
    # Check by modeler attributes (secondary method)
    # Memory-based: has fit() method and _fitted attribute
    if hasattr(modeler, 'fit') and hasattr(modeler, '_fitted'):
        return 'memory'
    
    # Check by model attributes
    if hasattr(modeler, 'model'):
        model = modeler.model
        if hasattr(model, 'gaussian') and hasattr(model, 'memory_bank'):
            # PaDiM pattern: has both gaussian distribution and memory bank
            return 'memory'
        elif hasattr(model, 'model_type'):
            model_type = model.model_type.lower()
            if any(name in model_type for name in MEMORY_BASED_MODELS):
                return 'memory'
            elif any(name in model_type for name in GRADIENT_BASED_MODELS):
                return 'gradient'
            elif any(name in model_type for name in FLOW_BASED_MODELS):
                return 'flow'
    
    # Default to gradient-based (most common)
    return 'gradient'


def infer_trainer_type_from_model_name(model_name):
    """Infer trainer type from model name string"""
    model_name = model_name.lower()
    
    if any(name in model_name for name in MEMORY_BASED_MODELS):
        return 'memory'
    elif any(name in model_name for name in GRADIENT_BASED_MODELS):
        return 'gradient'
    elif any(name in model_name for name in FLOW_BASED_MODELS):
        return 'flow'
    else:
        return 'gradient'  # default


def get_trainer(model_type_or_modeler, modeler=None, trainer_type=None, 
                scheduler=None, stopper=None, logger=None, **kwargs):
    """
    Factory function to create appropriate trainer based on model type
    
    Usage patterns:
        # Method 1: Auto-detect from modeler (recommended)
        trainer = get_trainer(modeler)
        
        # Method 2: Specify model name for explicit selection
        trainer = get_trainer('padim', modeler)
        
        # Method 3: Explicit trainer type override
        trainer = get_trainer(modeler, trainer_type='memory')
        
        # Method 4: With additional configuration
        trainer = get_trainer('ae', modeler, scheduler='plateau', stopper='early_stop')
    """
    
    # Handle different call patterns for backward compatibility
    if modeler is None:
        # Called as get_trainer(modeler, ...)
        actual_modeler = model_type_or_modeler
        actual_model_type = None
    else:
        # Called as get_trainer('model_name', modeler, ...)
        actual_modeler = modeler
        actual_model_type = model_type_or_modeler

    # Determine trainer type
    if trainer_type is not None:
        # Explicit override
        final_trainer_type = trainer_type
    elif actual_model_type is not None:
        # Based on model name
        final_trainer_type = infer_trainer_type_from_model_name(actual_model_type)
    else:
        # Auto-detect from modeler
        final_trainer_type = infer_trainer_type_from_modeler(actual_modeler)

    # Create appropriate trainer class
    trainer_classes = {
        'memory': MemoryTrainer,
        'gradient': GradientTrainer,
        'flow': FlowTrainer,
    }
    
    if final_trainer_type not in trainer_classes:
        raise ValueError(f"Unknown trainer type: {final_trainer_type}. Available: {list(trainer_classes.keys())}")

    trainer_class = trainer_classes[final_trainer_type]
    
    # Handle scheduler and stopper creation from strings
    if isinstance(scheduler, str):
        scheduler = get_scheduler(actual_modeler.configure_optimizers(), scheduler, **kwargs)
    if isinstance(stopper, str):
        stopper = get_stopper(stopper, **kwargs)

    return trainer_class(
        modeler=actual_modeler,
        scheduler=scheduler,
        stopper=stopper,
        logger=logger
    )


def create_trainer_with_config(modeler, config_dict):
    """Create trainer from configuration dictionary"""
    trainer_config = config_dict.get('trainer', {})
    
    trainer_type = trainer_config.get('type', None)
    scheduler_config = trainer_config.get('scheduler', None)
    stopper_config = trainer_config.get('stopper', None)
    
    scheduler = None
    if scheduler_config:
        if isinstance(scheduler_config, dict):
            scheduler = get_scheduler(
                modeler.configure_optimizers(),
                scheduler_config.get('name', 'plateau'),
                **scheduler_config.get('params', {})
            )
        else:
            scheduler = scheduler_config
    
    stopper = None
    if stopper_config:
        if isinstance(stopper_config, dict):
            stopper = get_stopper(
                stopper_config.get('name', 'early_stop'),
                **stopper_config.get('params', {})
            )
        else:
            stopper = stopper_config
    
    return get_trainer(
        modeler,
        trainer_type=trainer_type,
        scheduler=scheduler,
        stopper=stopper
    )


# Backward compatibility: create Trainer class that auto-detects type
class Trainer:
    """
    Backward compatibility wrapper that auto-detects trainer type
    
    This maintains the exact same interface as the original trainer.py:
        trainer = Trainer(modeler)
        history = trainer.fit(train_loader, num_epochs, valid_loader)
        scores, labels = trainer.predict(test_loader)
    """
    
    def __new__(cls, modeler, scheduler=None, stopper=None, logger=None):
        # Auto-detect and create appropriate trainer
        return get_trainer(
            modeler,
            scheduler=scheduler,
            stopper=stopper,
            logger=logger
        )


# Convenience functions for explicit trainer creation
def create_memory_trainer(modeler, logger=None, **kwargs):
    """Create memory trainer explicitly for PaDiM, PatchCore"""
    return MemoryTrainer(modeler, logger=logger, **kwargs)


def create_gradient_trainer(modeler, scheduler=None, stopper=None, logger=None, **kwargs):
    """Create gradient trainer explicitly for AE, STFPM"""
    return GradientTrainer(modeler, scheduler=scheduler, stopper=stopper, logger=logger, **kwargs)


def create_flow_trainer(modeler, scheduler=None, stopper=None, logger=None, **kwargs):
    """Create flow trainer explicitly for FastFlow, CFlow"""
    return FlowTrainer(modeler, scheduler=scheduler, stopper=stopper, logger=logger, **kwargs)


# Model type registry for easy extension
def register_model_type(model_name, trainer_type):
    """Register new model type with specific trainer type"""
    model_name = model_name.lower()
    trainer_type = trainer_type.lower()
    
    if trainer_type == 'memory':
        MEMORY_BASED_MODELS.add(model_name)
    elif trainer_type == 'gradient':
        GRADIENT_BASED_MODELS.add(model_name)
    elif trainer_type == 'flow':
        FLOW_BASED_MODELS.add(model_name)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


def get_registered_models():
    """Get all registered model types"""
    return {
        'memory': list(MEMORY_BASED_MODELS),
        'gradient': list(GRADIENT_BASED_MODELS),
        'flow': list(FLOW_BASED_MODELS),
    }


def diagnose_modeler(modeler):
    """Diagnostic function to understand modeler characteristics"""
    info = {
        'class_name': modeler.__class__.__name__,
        'has_fit_method': hasattr(modeler, 'fit'),
        'has_fitted_attribute': hasattr(modeler, '_fitted'),
        'learning_type': getattr(modeler, 'learning_type', 'unknown'),
        'detected_trainer_type': infer_trainer_type_from_modeler(modeler),
    }
    
    if hasattr(modeler, 'model'):
        model = modeler.model
        info.update({
            'model_class': model.__class__.__name__,
            'has_memory_bank': hasattr(model, 'memory_bank'),
            'has_gaussian': hasattr(model, 'gaussian'),
            'has_model_type': hasattr(model, 'model_type'),
        })
        
        if hasattr(model, 'model_type'):
            info['model_type'] = model.model_type
    
    return info


if __name__ == "__main__":
    # Test the factory system
    print("Trainer Factory System")
    print("=" * 40)
    
    print("\nRegistered Model Types:")
    registered = get_registered_models()
    for trainer_type, models in registered.items():
        print(f"  {trainer_type.capitalize()}: {models}")
    
    print(f"\nTrainer type inference examples:")
    print(f"  'padim' -> {infer_trainer_type_from_model_name('padim')}")
    print(f"  'ae' -> {infer_trainer_type_from_model_name('ae')}")
    print(f"  'fastflow' -> {infer_trainer_type_from_model_name('fastflow')}")
    print(f"  'unknown_model' -> {infer_trainer_type_from_model_name('unknown_model')}")