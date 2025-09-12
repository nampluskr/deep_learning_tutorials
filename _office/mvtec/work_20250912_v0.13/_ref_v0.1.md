### `main.py`

```python
import torchvision.transforms.v2 as v2

class TrainTransform:
    def __init__(self, img_size=256, **params):
        flip_prob = params.get('flip_prob', 0.5)
        rotation_degrees = params.get('rotation_degrees', 15)
        brightness = params.get('brightness', 0.1)
        contrast = params.get('contrast', 0.1)
        saturation = params.get('saturation', 0.1)
        hue = params.get('hue', 0.05)

        self.transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=15, fill=0),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            # v2.ToTensor(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestTransform:
    def __init__(self, img_size=256, **params):
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            #v2.ToTensor(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        return self.transform(image)

def get_data_for_gradient(categories=["bottle"]):
    return MVTecDataloader(
        data_dir=os.path.join(DATASET_DIR, "mvtec"),
        categories=categories,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        train_batch_size=4,
        test_batch_size=8,
        valid_ratio=0.2,
        num_workers=4,
        pin_memory=True
    )

def get_data_for_memory(categories=["bottle"]):
    return MVTecDataloader(
        data_dir='/home/namu/myspace/NAMU/datasets/mvtec',
        categories=categories,
        train_batch_size=4,
        test_batch_size=8,
        train_transform=TrainTransform(),
        test_transform=TestTransform(),
        valid_ratio=0.0,  # No validation for memory models
        num_workers=2,
        pin_memory=False,
    )

def run_stfpm(verbose=True):
    from models.model_stfpm import STFPMModel, STFPMLoss
    from modelers.modeler_stfpm import STFPMModeler
    from trainers.trainer_gradient import GradientTrainer
    from metrics.metrics_gradient import FeatureSimilarityMetric

    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: STFPM\n" + "="*50)
    categories=["grid"]
    # data = get_data_for_gradient(categories)
    data = get_data_for_memory(categories)

    modeler = STFPMModeler(
        model = STFPMModel(layers=["layer1", "layer2", "layer3"], backbone="resnet18"),
        loss_fn = STFPMLoss(),
        metrics = {"feature_sim": FeatureSimilarityMetric(similarity_fn='cosine')},
    )
    trainer = GradientTrainer(modeler, scheduler=None, stopper=None, logger=None)

    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)

    trainer.fit(data.train_loader(), num_epochs=100)
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)
```

### `dataset_mvtec.py`

```python
class MVTecDataset(Dataset):
    """MVTec AD dataset for anomaly detection"""

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return dict(image=image, label=label)
```

### Result

```
All backbone weights verified in: /home/namu/myspace/NAMU/project_2025/backbones

==================================================
RUNNING EXPERIMENT: STFPM
==================================================
/opt/conda/lib/python3.10/site-packages/torchvision/transforms/v2/_deprecated.py:41: UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.
  warnings.warn(

 > Dataset Type:      /home/namu/myspace/NAMU/datasets/mvtec
 > Categories:        ['grid']
 > Train data:        264
 > Valid data:        None (no validation split)
 > Test data:         78

 > Modeler Type:      STFPMModeler
 > Model Type:        STFPMModel
 > Total params.:     5,565,568
 > Trainable params.: 2,782,784
 > Learning Type:     one_class
 > Loss Function:     STFPMLoss
 > Metrics:           ['feature_sim']
 > Device:            cuda

 > Trainer Type:      gradient
 > Optimizer:         AdamW
 > Learning Rate:     0.0001
 > Scheduler:         None
 > Stopper:           None
 > Logger:            None

 > Training started...
 [ 1/100] loss=2.069, feature_sim=0.828 (5.9s)                                                            
 [ 2/100] loss=1.170, feature_sim=0.903 (3.6s)                                                            
 [ 3/100] loss=0.909, feature_sim=0.924 (3.6s)                                                            
 [ 4/100] loss=0.761, feature_sim=0.937 (3.5s)
... 
 [93/100] loss=0.228, feature_sim=0.981 (4.5s)                                                            
 [94/100] loss=0.226, feature_sim=0.981 (4.2s)                                                            
 [95/100] loss=0.221, feature_sim=0.982 (4.9s)                                                            
 [96/100] loss=0.220, feature_sim=0.982 (4.4s)                                                            
 [97/100] loss=0.215, feature_sim=0.982 (4.2s)                                                            
 [98/100] loss=0.213, feature_sim=0.982 (3.8s)                                                            
 [99/100] loss=0.220, feature_sim=0.982 (3.6s)                                                            
 [100/100] loss=0.218, feature_sim=0.982 (3.7s)                                                           
 > Training completed!
------------------------------------------------------------                                              
EXPERIMENT RESULTS
------------------------------------------------------------
 > AUROC:             0.8129
 > AUPR:              0.9361
 > Threshold:         9.143e-06
------------------------------------------------------------
 > Accuracy:          0.7821
 > Precision:         0.9348
 > Recall:            0.7544
 > F1-Score:          0.8350
------------------------------------------------------------
                   Predicted
   Actual    Normal  Anomaly
   Normal        18        3
   Anomaly       14       43
```
