from model_ae import VanillaAE, UNetAE
from model_ae import MSELoss, BCELoss, CombinedLoss
from model_ae import PSNRMetric, SSIMMetric


def get_model(name, **params):
    available_list = {
        "ae": VanillaAE,
        "vanilla_ae": VanillaAE,
        "unet_ae": UNetAE,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(**default_params)


def get_loss(name, **params):
    available_list = {
        "mse": MSELoss,
        "bce": BCELoss,
        "combined": CombinedLoss,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(**default_params)


def get_metric(name, **params):
    available_list = {
        "psnr": PSNRMetric,
        "ssim": SSIMMetric,
        # "lpips": LPIPSMetric,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(**default_params)



if __name__ == "__main__":
    pass
