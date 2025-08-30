import torch
from torch import optim
from collections.abc import Sequence

from modeler_base import BaseModeler
from model_stfpm import STFPMModel, STFPMLoss


class STFPMModeler(BaseModeler):
    
    def __init__(
        self,
        backbone="resnet18",
        layers=("layer1", "layer2", "layer3"),
        metrics=None,
        device=None
    ):
        model = STFPMModel(backbone=backbone, layers=layers)
        loss_fn = STFPMLoss()
        
        super().__init__(model, loss_fn, metrics, device)
        
        self.backbone = backbone
        self.layers = layers
    
    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)
        
        optimizer.zero_grad()
        teacher_features, student_features = self.model.forward(inputs['image'])
        
        loss = self.loss_fn(teacher_features, student_features)
        
        loss.backward()
        optimizer.step()
        
        results = {'loss': loss.item()}
        
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                metric_value = metric_fn(teacher_features, student_features)
                results[metric_name] = float(metric_value)
        
        return results
    
    @torch.no_grad()
    def validate_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)
        
        predictions = self.model(inputs['image'])
        
        if hasattr(predictions, 'anomaly_map'):
            anomaly_maps = predictions.anomaly_map
            pred_scores = predictions.pred_score
        else:
            anomaly_maps = predictions
            pred_scores = self.compute_image_scores(anomaly_maps)
        
        loss = 0.0
        try:
            self.model.train()
            teacher_features, student_features = self.model(inputs['image'])
            loss = self.loss_fn(teacher_features, student_features).item()
            self.model.eval()
        except Exception:
            pass
        
        results = {'loss': loss}
        
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn(pred_scores, inputs['image'])
            results[metric_name] = float(metric_value)
        
        return results
    
    @torch.no_grad()
    def predict_step(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Perform prediction step for STFPM.
        
        Generates anomaly maps and returns image-level prediction scores.
        
        Args:
            inputs (Dict[str, Any]): Input batch containing 'image'
            
        Returns:
            torch.Tensor: Image-level anomaly prediction scores
        """
        self.model.eval()
        inputs = self.to_device(inputs)
        
        # Generate predictions
        predictions = self.model(inputs['image'])
        
        # Extract prediction scores
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback - compute scores from anomaly maps
            return self.compute_image_scores(predictions)
    
    def compute_anomaly_scores(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute both pixel-level and image-level anomaly scores.
        
        Args:
            inputs (Dict[str, Any]): Input batch containing 'image'
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'anomaly_maps' and 'pred_scores'
        """
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            predictions = self.model(inputs['image'])
            
            if hasattr(predictions, 'anomaly_map'):
                # InferenceBatch format
                return {
                    'anomaly_maps': predictions.anomaly_map,
                    'pred_scores': predictions.pred_score
                }
            else:
                # Direct anomaly maps
                return {
                    'anomaly_maps': predictions,
                    'pred_scores': self.compute_image_scores(predictions)
                }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for STFPM training.
        
        Uses the exact same optimizer configuration as the original Lightning implementation:
        - SGD optimizer targeting only student model parameters
        - Learning rate: 0.4, Momentum: 0.9, Weight decay: 0.001
        
        Returns:
            torch.optim.Optimizer: Configured SGD optimizer
        """
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )
    
    @property
    def learning_type(self) -> str:
        """
        Get the learning type of the model.
        
        Returns:
            str: "one_class" - STFPM uses one-class learning
        """
        return "one_class"
    
    @property
    def trainer_arguments(self) -> Dict[str, Any]:
        """
        Get trainer arguments specific to STFPM.
        
        Uses the exact same trainer arguments as the original Lightning implementation.
        
        Returns:
            Dict[str, Any]: Dictionary of trainer arguments:
                - gradient_clip_val: 0 (disable gradient clipping)
                - num_sanity_val_steps: 0 (skip validation sanity checks)
        """
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }
    
    def get_teacher_features(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract features from teacher network only.
        
        Args:
            inputs (Dict[str, Any]): Input batch containing 'image'
            
        Returns:
            Dict[str, torch.Tensor]: Teacher network features
        """
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            return self.model.teacher_model(inputs['image'])
    
    def get_student_features(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract features from student network only.
        
        Args:
            inputs (Dict[str, Any]): Input batch containing 'image'
            
        Returns:
            Dict[str, torch.Tensor]: Student network features
        """
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            return self.model.student_model(inputs['image'])


# Factory function for compatibility
def get_stfpm_modeler(
    backbone: str = "resnet18",
    layers: Sequence[str] = ("layer1", "layer2", "layer3"), 
    metrics: Dict[str, Any] = None,
    device: str = None
) -> STFPMModeler:
    """
    Factory function to create STFPM modeler.
    
    Args:
        backbone (str): Backbone architecture name
        layers (Sequence[str]): Feature extraction layers  
        metrics (Dict[str, Any]): Dictionary of metric functions
        device (str): Device to run on
        
    Returns:
        STFPMModeler: Configured STFPM modeler instance
    """
    return STFPMModeler(
        backbone=backbone,
        layers=layers,
        metrics=metrics,
        device=device
    )


if __name__ == "__main__":
    # Test the modeler
    print("Testing STFPM Modeler...")
    
    modeler = STFPMModeler()
    
    # Test input
    inputs = {
        'image': torch.randn(2, 3, 256, 256),
        'label': torch.tensor([0, 1])
    }
    
    print(f"Device: {modeler.device}")
    print(f"Learning type: {modeler.learning_type}")
    print(f"Trainer arguments: {modeler.trainer_arguments}")
    
    # Test optimizer configuration
    optimizer = modeler.configure_optimizers()
    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Momentum: {optimizer.param_groups[0]['momentum']}")
    
    # Test training step
    train_results = modeler.train_step(inputs, optimizer)
    print(f"Train results: {train_results}")
    
    # Test validation step  
    val_results = modeler.validate_step(inputs)
    print(f"Validation results: {val_results}")
    
    # Test prediction step
    scores = modeler.predict_step(inputs)
    print(f"Prediction scores shape: {scores.shape}")
    
    # Test anomaly score computation
    anomaly_results = modeler.compute_anomaly_scores(inputs)
    print(f"Anomaly maps shape: {anomaly_results['anomaly_maps'].shape}")
    print(f"Pred scores shape: {anomaly_results['pred_scores'].shape}")
    
    print("STFPM Modeler test completed!")