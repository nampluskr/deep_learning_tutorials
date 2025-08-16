from config import Config, print_config
from train import train_model


if __name__ == "__main__":

    config1 = Config(
        model_type='unet_ae',
        num_epochs=5,
    )
    config2 = Config(
        model_type='vanilla_ae',
        num_epochs=5,
    )
    config_list = [config1, config2]

    for idx, config in enumerate(config_list):
        print(f"\n*** Training model [{idx + 1}/{len(config_list)}]")
        print_config(config)
        train_model(config)
