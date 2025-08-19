import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv_block(x)


class VanillaEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()

        # 점진적 다운샘플링: 256x256 -> 8x8
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 64),         # 256x256 -> 128x128
            ConvBlock(64, 128),                 # 128x128 -> 64x64
            ConvBlock(128, 256),                # 64x64 -> 32x32
            ConvBlock(256, 512),                # 32x32 -> 16x16
            ConvBlock(512, 512)                 # 16x16 -> 8x8
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.conv_blocks(x)              # [B, 512, 8, 8]
        pooled = self.global_pool(features)         # [B, 512, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)    # [B, 512]
        latent = self.fc(pooled)                    # [B, latent_dim]

        return latent, features


class VanillaDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=512):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (512, 8, 8))

        # 점진적 업샘플링: 8x8 -> 256x256
        self.deconv_blocks = nn.Sequential(
            DeconvBlock(512, 512),                          # 8x8 -> 16x16
            DeconvBlock(512, 256),                          # 16x16 -> 32x32
            DeconvBlock(256, 128),                          # 32x32 -> 64x64
            DeconvBlock(128, 64),                           # 64x64 -> 128x128
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),  # 128x128 -> 256x256
            nn.Sigmoid()                                    # [0, 1] 범위로 정규화
        )

    def forward(self, latent):
        x = self.fc(latent)                     # [B, 512*8*8]
        x = self.unflatten(x)                   # [B, 512, 8, 8]
        reconstructed = self.deconv_blocks(x)   # [B, 3, 256, 256]
        return reconstructed


class VanillaAutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = VanillaEncoder(in_channels, latent_dim)
        self.decoder = VanillaDecoder(out_channels, latent_dim)

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        latent, features = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, features

    def encode(self, x):
        latent, features = self.encoder(x)
        return latent, features

    def decode(self, latent):
        reconstructed = self.decoder(latent)
        return reconstructed


# 테스트용 함수
def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VanillaAutoEncoder(latent_dim=512).to(device)

    # 더미 입력 생성
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        reconstructed, latent, features = model(dummy_input)

    # 결과 출력
    print("="*50)
    print("Vanilla AutoEncoder 테스트")
    print("="*50)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Features shape: {features.shape}")

    # 모델 정보 출력
    model_info = model.get_model_info()
    print(f"\n모델 정보:")
    for key, value in model_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    return model


if __name__ == "__main__":

    test_model()