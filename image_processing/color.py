import numpy as np
import matplotlib.pyplot as plt


def to_image(sRGB):
    """ sRGB[0, 1] to 8bit RGB[0, 255] """
    sRGB = np.clip(sRGB, 0.0, 1.0)
    return np.round(sRGB * 255.0).astype(np.uint8)


def to_sRGB(RGB8bit):
    """ 8bit RGB[0, 255] to sRGB[0, 1] """
    return RGB8bit.astype(np.float32) / 255.0


def merge(X, Y, Z):
    """ [height, width] to [height, width, 3] """
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def sRGB_to_linRGB(sRGB):
    """ sRGB[0, 1] to linear RGB[0, 1] """
    sRGB = np.clip(sRGB, 0.0, 1.0).astype(np.float32)
    mask = sRGB <= 0.04045
    linRGB = np.empty_like(sRGB)
    linRGB[mask] = sRGB[mask] / 12.92
    linRGB[~mask] = np.power((sRGB[~mask] + 0.055)/1.055, 2.4)
    return linRGB


def linRGB_to_sRGB(linRGB):
    """ linear RGB[0, 1] to sRGB[0, 1] """
    linRGB = np.clip(linRGB, 0.0, 1.0).astype(np.float32)
    mask = linRGB <= 0.0031308
    sRGB = np.empty_like(linRGB)
    sRGB[mask] = linRGB[mask] * 12.92
    sRGB[~mask] = 1.055 * np.power(linRGB[~mask], 1.0/2.4) - 0.055
    return sRGB


def apply_dimming(sRGB, dimming):
    linRGB = sRGB_to_linRGB(sRGB)
    linRGB_dimmed = np.clip(linRGB*dimming, 0.0, 1.0)
    sRGB = linRGB_to_sRGB(linRGB_dimmed)
    return sRGB


def get_RGB2XYZ(Rx=0.64, Ry=0.33, Gx=0.30, Gy=0.60, 
                Bx=0.15, By=0.06, Wx=0.3127, Wy=0.3290):
    red    = np.r_[Rx, Ry, 1 - Rx - Ry] / Ry
    green  = np.r_[Gx, Gy, 1 - Gx - Gy] / Gy
    blue   = np.r_[Bx, By, 1 - Bx - By] / By
    white  = np.r_[Wx, Wy, 1 - Wx - Wy] / Wy

    matrix = np.vstack([red, green, blue]).astype(np.float32).T
    scale  = np.linalg.solve(matrix, white).astype(np.float32)
    return matrix * scale


def sRGB_to_XYZ(sRGB):
    linRGB = sRGB_to_linRGB(sRGB)
    RGB2XYZ = get_RGB2XYZ()
    return linRGB.dot(RGB2XYZ.T)


def XYZ_to_sRGB(XYZ):
    RGB2XYZ = get_RGB2XYZ()
    XYZ2RGB = np.linalg.inv(RGB2XYZ)
    linRGB = XYZ.dot(XYZ2RGB.T)
    return linRGB_to_sRGB(linRGB)


def XYZ_to_Lxy(XYZ, eps=1e-8):
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    XYZ_sum = X + Y + Z
    x = np.where(XYZ_sum > eps, X / XYZ_sum, 0.0)
    y = np.where(XYZ_sum > eps, Y / XYZ_sum, 0.0)
    return np.stack([Y, x, y], axis=-1).astype(np.float32)


if __name__ == "__main__":
    
    pass
