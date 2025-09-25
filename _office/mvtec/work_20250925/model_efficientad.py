import os
from collections.abc import Sequence
from tqdm import tqdm
from time import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from feature_extractor import TimmFeatureExtractor
from trainer import BaseTrainer


def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    return (x - mean) / std


def reduce_tensor_elems(tensor: torch.Tensor, m: int = 2**24) -> torch.Tensor:
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class SmallPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels, padding=False):
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class MediumPatchDescriptionNetwork(nn.Module):
    def __init__(self, out_channels, padding=False):
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


###########################################################
# Autoencoder for EfficientAD Model
###########################################################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = padding
        # use ceil to match output shape of PDN
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x, image_size):
        last_upsample = (
            math.ceil(image_size[0] / 4) if self.padding else math.ceil(image_size[0] / 4) - 8,
            math.ceil(image_size[1] / 4) if self.padding else math.ceil(image_size[1] / 4) - 8,
        )
        x = F.interpolate(x, size=(image_size[0] // 64 - 1, image_size[1] // 64 - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(image_size[0] // 32, image_size[1] // 32), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(image_size[0] // 16 - 1, image_size[1] // 16 - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(image_size[0] // 8, image_size[1] // 8), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(image_size[0] // 4 - 1, image_size[1] // 4 - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(image_size[0] // 2 - 1, image_size[1] // 2 - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    def __init__(self, out_channels, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x, image_size):
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        return self.decoder(x, image_size)


###########################################################
# EfficientAD Model
###########################################################

class EfficientAD(nn.Module):
    def __init__(self, teacher_out_channels=384, model_size="small",
        image_shape=(256, 256), padding=False, pad_maps=True):
        super().__init__()
        self.model_size = model_size
        self.image_shape = image_shape
        self.pad_maps = pad_maps

        if model_size == "medium":
            self.teacher = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = MediumPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)
        elif model_size == "small":
            self.teacher = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = SmallPatchDescriptionNetwork(out_channels=teacher_out_channels * 2, padding=padding)
        else:
            raise ValueError(f"Unknown model size {model_size}")

        self.ae = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels = teacher_out_channels

        self.mean_std: nn.ParameterDict = nn.ParameterDict({
            "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),},)

        self.quantiles: nn.ParameterDict = nn.ParameterDict({
            "qa_st": torch.tensor(0.0),
            "qb_st": torch.tensor(0.0),
            "qa_ae": torch.tensor(0.0),
            "qb_ae": torch.tensor(0.0),},)

    @staticmethod
    def is_set(p_dic):
        return any(value.sum() != 0 for _, value in p_dic.items())

    @staticmethod
    def choose_random_aug_image(image):
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        coefficient = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
        idx = int(torch.randint(0, len(transform_functions), (1,)).item())
        transform_function = transform_functions[idx]
        return transform_function(image, coefficient)

    def forward(self, batch, batch_imagenet=None, normalize=True):
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        return self.compute_losses(batch, batch_imagenet, distance_st)

    def predict(self, batch, normalize=True):
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        map_st, map_stae = self.compute_maps(batch, student_output, distance_st, normalize)
        anomaly_map = 0.5 * map_st + 0.5 * map_stae
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_student_teacher_distance(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)
        return student_output, distance_st

    def compute_losses(self, batch, batch_imagenet, distance_st):
        # Student loss
        distance_st = reduce_tensor_elems(distance_st)
        d_hard = torch.quantile(distance_st, 0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        penalty_input = batch_imagenet if batch_imagenet is not None else batch
        student_output_penalty = self.student(penalty_input)[:, : self.teacher_out_channels, :, :]
        loss_penalty = torch.mean(student_output_penalty ** 2)
        loss_st = loss_hard + loss_penalty

        # Autoencoder and Student AE Loss
        aug_img = self.choose_random_aug_image(batch)
        ae_output_aug = self.ae(aug_img, batch.shape[-2:])

        with torch.no_grad():
            teacher_output_aug = self.teacher(aug_img)
            if self.is_set(self.mean_std):
                teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

        student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]
        distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
        distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        return loss_st, loss_ae, loss_stae

    def compute_maps(self, batch, student_output, distance_st, normalize=True):
        image_size = batch.shape[-2:]
        with torch.no_grad():
            ae_output = self.ae(batch, image_size)
            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean((ae_output - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1, keepdim=True)

        if self.pad_maps:
            map_st = F.pad(map_st, (4, 4, 4, 4))
            map_stae = F.pad(map_stae, (4, 4, 4, 4))

        map_st = F.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = F.interpolate(map_stae, size=image_size, mode="bilinear")

        if self.is_set(self.quantiles) and normalize:
            map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
            map_stae = 0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
        return map_st, map_stae

    def get_maps(self, batch, normalize=False):
        student_output, distance_st = self.compute_student_teacher_distance(batch)
        return self.compute_maps(batch, student_output, distance_st, normalize)


###########################################################
# Trainer for EfficientAD Model
###########################################################

class EfficientADTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, metrics=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        trainable = list(self.model.student.parameters()) + list(self.model.ae.parameters())
        self.optimizer = optim.Adam(trainable, lr=1e-3, weight_decay=1e-5)

        self.batch_size = 1
        self.prepare_pretrained_model(self.model.model_size)
        self.prepare_imagenette_data(self.model.image_shape)

    def prepare_pretrained_model(self, model_size):
        weight_dir = "/home/namu/myspace/NAMU/project_2025/backbones"
        if not os.path.isdir(weight_dir):
            raise RuntimeError(f"Pretrained weight directory not found: {weight_dir}")

        teacher_path = os.path.join(weight_dir, f"pretrained_teacher_{model_size}.pth")
        if not os.path.isfile(teacher_path):
            raise RuntimeError(f"Teacher weight file not found: {teacher_path}")

        print(f" > Load pretrained teacher model from {teacher_path}")
        state = torch.load(teacher_path, map_location=self.device, weights_only=True)
        self.model.teacher.load_state_dict(state)

    def prepare_imagenette_data(self, image_shape):
        imagenet_dir = "/home/namu/myspace/NAMU/project_2025/backbones/imagenette2"
        if not os.path.isdir(imagenet_dir):
            raise RuntimeError(f"Imagenette directory not found: {imagenet_dir}")

        transforms = T.Compose([
            T.Resize((image_shape[0]*2, image_shape[1]*2)),
            T.RandomGrayscale(p=0.3),
            T.CenterCrop((image_shape[0], image_shape[1])),
            # T.ToImage(),
            # T.ToDtype(torch.float32, scale=True),
            T.ToTensor(),
        ])
        imagenet_dataset = ImageFolder(imagenet_dir, transform=transforms)
        self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.imagenet_iterator = iter(self.imagenet_loader)

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader):
        arrays_defined = False
        n: torch.Tensor | None = None
        chanel_sum: torch.Tensor | None = None
        chanel_sum_sqr: torch.Tensor | None = None

        for batch in tqdm(dataloader, desc="Calculate mean & std", ascii=True, position=0, leave=False):
            y = self.model.teacher(batch["image"].to(self.device))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                chanel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                chanel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                arrays_defined = True

            n += y[:, 0].numel()
            chanel_sum += torch.sum(y, dim=[0, 2, 3])
            chanel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])

        if n is None:
            msg = "The value of 'n' cannot be None."
            raise ValueError(msg)

        channel_mean = chanel_sum / n
        channel_std = (torch.sqrt((chanel_sum_sqr / n) - (channel_mean**2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]
        return {"mean": channel_mean, "std": channel_std}

    def map_norm_quantiles(self, loader):
        maps_st = []
        maps_ae = []
        for batch in tqdm(loader, desc="Calculate Quantiles", ascii=True, position=0, leave=False):
            for img, label in zip(batch["image"], batch["label"], strict=True):
                if label == 0:  # only use good images of validation set!
                    map_st, map_ae = self.model.get_maps(img.to(self.device), normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)

        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps):
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    def run_epoch(self, loader, mode="train", desc=""):
        stats = {"loss": 0.0, "st": 0.0, "ae": 0.0, "stae": 0.0}
        num_images = 0

        with tqdm(loader, desc=desc, ascii=True, leave=False) as pbar:
            for batch in pbar:
                imgs = batch["image"].to(self.device)
                batch_size = imgs.size(0)
                num_images += batch_size

                try:
                    imgnet = next(self.imagenet_iterator)[0].to(self.device)
                except StopIteration:
                    self.imagenet_iterator = iter(self.imagenet_loader)
                    imgnet = next(self.imagenet_iterator)[0].to(self.device)

                loss_st, loss_ae, loss_stae = self.model(batch=imgs, batch_imagenet=imgnet)
                loss = loss_st + loss_ae + loss_stae

                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                stats["loss"] += loss.item() * batch_size
                stats["st"]   += loss_st.item() * batch_size
                stats["ae"]   += loss_ae.item() * batch_size
                stats["stae"] += loss_stae.item() * batch_size

                pbar.set_postfix({
                    "loss": f"{stats['loss']/num_images:.3}",
                    "st":   f"{stats['st']/num_images:.3}",
                    "ae":   f"{stats['ae']/num_images:.3f}",
                    "stae": f"{stats['stae']/num_images:.3f}",
                })
        return {k: v / num_images for k, v in stats.items()}

    def fit(self, train_loader, num_epochs, valid_loader=None, weight_path=None):
        train_start_time = time()
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time()
            mean_std = self.teacher_channel_mean_std(train_loader)
            self.model.mean_std.update(mean_std)

            train_results = self.train(train_loader, epoch, num_epochs)
            train_info = ", ".join([f'{key}={value:.2e}' for key, value in train_results.items()])
            print(f" [{epoch:3d}/{num_epochs}] {train_info} ({time() - epoch_start_time:.1f}s)")

            if valid_loader is not None:
                quantiles = self.map_norm_quantiles(valid_loader)
                self.model.quantiles.update(quantiles)

                # valid_results = self.validate(train_loader, epoch, num_epochs)
                # valid_info = ", ".join([f'{key}={value:.2e}' for key, value in valid_results.items()])
                # print(f" [{epoch:3d}/{num_epochs}] {train_info} | (val) {valid_info} ({time() - epoch_start_time:.1f}s)")

                if epoch % 1 == 0:
                    eval_img = self.evaluate_image_level(valid_loader, method="f1")
                    img_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_img.items() if isinstance(v, float)])
                    img_info2 = ", ".join([f"{k}={v:2d}" for k, v in eval_img.items() if isinstance(v, int)])
                    print(f" > Image-level: {img_info1} | {img_info2}\n")
            else:
                print(f" [{epoch:3d}/{num_epochs}] {train_info} ({time() - epoch_start_time:.1f}s)")

        elapsed_time = time() - train_start_time
        hours, reminder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(reminder, 60)
        print(f" > Training finished... in {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        self.save_model(weight_path)
        # return history


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientAD(model_size="small").to(device)
    model.eval()
    input_tensor = torch.randn(32, 3, 256, 256).to(device)

    loss_st, loss_ae, loss_stae = model(input_tensor)
    print(f"{loss_st.item():.3f}, {loss_ae.item():.3f}, {loss_stae.item():.3f}")

    predictions = model.predict(input_tensor)
    print(predictions['anomaly_map'].shape)     # torch.Size([32, 1, 256, 256])
    print(predictions['pred_score'].shape)      # torch.Size([32, 1])


