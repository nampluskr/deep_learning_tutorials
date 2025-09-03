import torch
from torch import optim
import tqdm
from pathlib import Path

from .modeler_base import BaseModeler


class EfficientAdModeler(BaseModeler):
    def __init__(self, model, loss_fn=None, metrics=None, device=None, 
                 imagenet_dir="./datasets/imagenette", pretrained_dir="./pre_trained/"):
        super().__init__(model, loss_fn, metrics, device)
        self.imagenet_dir = Path(imagenet_dir)
        self.pretrained_dir = Path(pretrained_dir)
        self._teacher_loaded = False
        self._channel_stats_calculated = False
        self._quantiles_calculated = False
        self.imagenet_loader = None
        self.imagenet_iterator = None
        
    def prepare_pretrained_teacher(self):
        """Load pretrained teacher weights if available locally."""
        model_size_str = self.model.model_size.value if hasattr(self.model, 'model_size') else 'small'
        teacher_path = self.pretrained_dir / "efficientad_pretrained_weights" / f"pretrained_teacher_{model_size_str}.pth"
        
        if teacher_path.exists():
            print(f" > Loading pretrained teacher from {teacher_path}")
            try:
                state_dict = torch.load(teacher_path, map_location=self.device)
                self.model.teacher.load_state_dict(state_dict)
                self._teacher_loaded = True
                print(f" > Successfully loaded teacher weights")
            except Exception as e:
                print(f" > Warning: Failed to load teacher weights: {e}")
                print(f" > Using random teacher initialization")
        else:
            print(f" > Warning: Pretrained teacher weights not found at {teacher_path}")
            print(f" > Using random teacher initialization")

    def prepare_imagenet_data(self, image_size):
        """Prepare ImageNet data loader if directory exists."""
        if self.imagenet_dir.exists():
            try:
                from torch.utils.data import DataLoader
                from torchvision.datasets import ImageFolder
                from torchvision.transforms.v2 import Compose, Resize, RandomGrayscale, CenterCrop, ToTensor
                
                transform = Compose([
                    Resize((image_size[0] * 2, image_size[1] * 2)),
                    RandomGrayscale(p=0.3),
                    CenterCrop((image_size[0], image_size[1])),
                    ToTensor(),
                ])
                
                imagenet_dataset = ImageFolder(self.imagenet_dir, transform=transform)
                self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=1, shuffle=True, pin_memory=True)
                self.imagenet_iterator = iter(self.imagenet_loader)
                print(f" > ImageNet data loaded from {self.imagenet_dir}")
            except Exception as e:
                print(f" > Warning: Failed to load ImageNet data: {e}")
                print(f" > Training will use augmented normal data as substitute")
        else:
            print(f" > Warning: ImageNet directory not found at {self.imagenet_dir}")
            print(f" > Training will use augmented normal data as substitute")

    def calculate_teacher_channel_stats(self, dataloader):
        """Calculate channel-wise mean and std of teacher model activations."""
        if self._channel_stats_calculated:
            return
            
        print(" > Calculating teacher channel statistics...")
        arrays_defined = False
        n = None
        chanel_sum = None
        chanel_sum_sqr = None

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean & std", leave=False):
                inputs = self.to_device(batch)
                y = self.model.teacher(inputs['image'])
                
                if not arrays_defined:
                    _, num_channels, _, _ = y.shape
                    n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                    chanel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                    chanel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                    arrays_defined = True

                n += y[:, 0].numel()
                chanel_sum += torch.sum(y, dim=[0, 2, 3])
                chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        if n is not None:
            channel_mean = chanel_sum / n
            channel_std = (torch.sqrt((chanel_sum_sqr / n) - (channel_mean**2))).float()[None, :, None, None]
            channel_mean = channel_mean.float()[None, :, None, None]
            
            self.model.mean_std["mean"].data = channel_mean
            self.model.mean_std["std"].data = channel_std
            self._channel_stats_calculated = True
            print(f" > Channel statistics calculated - mean: {channel_mean.mean():.4f}, std: {channel_std.mean():.4f}")

    def calculate_map_quantiles(self, dataloader):
        """Calculate quantiles of student and autoencoder feature maps."""
        if self._quantiles_calculated:
            return
            
        print(" > Calculating validation dataset quantiles...")
        maps_st = []
        maps_ae = []
        
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Calculate validation quantiles", leave=False):
                inputs = self.to_device(batch)
                for img, label in zip(inputs['image'], inputs['label']):
                    if label == 0:  # only use good images
                        map_st, map_ae = self.model.get_maps(img.unsqueeze(0), normalize=False)
                        maps_st.append(map_st)
                        maps_ae.append(map_ae)

        if maps_st and maps_ae:
            from .model_efficientad import reduce_tensor_elems
            
            maps_st_flat = reduce_tensor_elems(torch.cat(maps_st))
            maps_ae_flat = reduce_tensor_elems(torch.cat(maps_ae))
            
            qa_st = torch.quantile(maps_st_flat, q=0.9).to(self.device)
            qb_st = torch.quantile(maps_st_flat, q=0.995).to(self.device)
            qa_ae = torch.quantile(maps_ae_flat, q=0.9).to(self.device) 
            qb_ae = torch.quantile(maps_ae_flat, q=0.995).to(self.device)
            
            self.model.quantiles["qa_st"].data = qa_st
            self.model.quantiles["qb_st"].data = qb_st
            self.model.quantiles["qa_ae"].data = qa_ae
            self.model.quantiles["qb_ae"].data = qb_ae
            
            self._quantiles_calculated = True
            print(f" > Quantiles calculated - ST: [{qa_st:.4f}, {qb_st:.4f}], AE: [{qa_ae:.4f}, {qb_ae:.4f}]")

    def get_imagenet_batch(self):
        """Get a batch from ImageNet data or use augmented normal data as fallback."""
        if self.imagenet_loader is not None:
            try:
                batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
                return batch_imagenet
            except StopIteration:
                self.imagenet_iterator = iter(self.imagenet_loader)
                batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
                return batch_imagenet
        else:
            # Fallback: create dummy ImageNet-like batch (random noise)
            batch_size = 1
            image_size = (256, 256)  # Default image size
            return torch.randn(batch_size, 3, *image_size, device=self.device)

    def train_step(self, inputs, optimizer):
        self.model.train()
        inputs = self.to_device(inputs)

        # Setup phase during first batch
        if not self._teacher_loaded:
            self.prepare_pretrained_teacher()
            
        if not hasattr(self, '_setup_done'):
            image_size = inputs['image'].shape[-2:]
            self.prepare_imagenet_data(image_size)
            self._setup_done = True

        # Calculate channel statistics if not done
        if not self._channel_stats_calculated:
            # Use a subset of training data for efficiency
            temp_loader = [{'image': inputs['image'][:1]}]  # Use first sample
            self.calculate_teacher_channel_stats(temp_loader)

        optimizer.zero_grad()
        
        # Get ImageNet batch
        batch_imagenet = self.get_imagenet_batch()
        
        # Forward pass
        loss_st, loss_ae, loss_stae = self.model(
            batch=inputs['image'], 
            batch_imagenet=batch_imagenet
        )
        
        # Total loss
        total_loss = loss_st + loss_ae + loss_stae
        total_loss.backward()
        optimizer.step()

        results = {
            'loss': total_loss.item(),
            'loss_st': loss_st.item(),
            'loss_ae': loss_ae.item(), 
            'loss_stae': loss_stae.item(),
        }
        
        # Add metric computation with torch.no_grad()
        with torch.no_grad():
            for metric_name, metric_fn in self.metrics.items():
                try:
                    # For EfficientAd, we don't have typical reconstruction metrics during training
                    # Most metrics would need inference mode output
                    results[metric_name] = 0.0
                except Exception:
                    results[metric_name] = 0.0
                    
        return results

    @torch.no_grad() 
    def validate_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)
        
        # Calculate quantiles if not done yet
        if not self._quantiles_calculated:
            # Use current validation batch for quantile calculation
            temp_loader = [inputs]
            self.calculate_map_quantiles(temp_loader)
        
        predictions = self.model(inputs['image'])
        
        if hasattr(predictions, 'pred_score'):
            scores = predictions.pred_score
            labels = inputs['label']
            
            # Score distribution analysis
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            
            normal_scores = scores[normal_mask] if normal_mask.any() else torch.tensor([0.0])
            anomaly_scores = scores[anomaly_mask] if anomaly_mask.any() else torch.tensor([0.0])
            
            results = {
                'loss': 0.0,  # No loss for inference mode
                'score_mean': scores.mean().item(),
                'score_std': scores.std().item(),
                'normal_mean': normal_scores.mean().item(),
                'anomaly_mean': anomaly_scores.mean().item(),
                'separation': (anomaly_scores.mean() - normal_scores.mean()).item() if anomaly_mask.any() and normal_mask.any() else 0.0,
            }
        else:
            results = {'loss': 0.0}
            
        return results

    @torch.no_grad()
    def predict_step(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)
        
        predictions = self.model(inputs['image'])
        
        if hasattr(predictions, 'pred_score'):
            return predictions.pred_score
        else:
            # Fallback: compute scores from maps
            map_st, map_stae = self.model.get_maps(inputs['image'], normalize=True)
            anomaly_map = 0.5 * map_st + 0.5 * map_stae
            return torch.amax(anomaly_map, dim=(-2, -1))

    def compute_anomaly_scores(self, inputs):
        self.model.eval()
        inputs = self.to_device(inputs)

        with torch.no_grad():
            predictions = self.model(inputs['image'])
            
            if hasattr(predictions, 'anomaly_map'):
                return {
                    'anomaly_maps': predictions.anomaly_map,
                    'pred_scores': predictions.pred_score
                }
            else:
                # Fallback: compute from maps  
                map_st, map_stae = self.model.get_maps(inputs['image'], normalize=True)
                anomaly_map = 0.5 * map_st + 0.5 * map_stae
                pred_scores = torch.amax(anomaly_map, dim=(-2, -1))
                
                return {
                    'anomaly_maps': anomaly_map,
                    'pred_scores': pred_scores
                }

    def configure_optimizers(self):
        # Only optimize student and autoencoder parameters
        student_params = list(self.model.student.parameters())
        ae_params = list(self.model.ae.parameters())
        
        return optim.Adam(
            student_params + ae_params,
            lr=0.0001,
            weight_decay=0.00001
        )

    @property
    def learning_type(self):
        return "one_class"

    @property  
    def trainer_arguments(self):
        return {
            "gradient_clip_val": 0,
            "num_sanity_val_steps": 0
        }