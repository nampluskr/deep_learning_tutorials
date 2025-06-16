```python
import numpy as np
import matplotlib.pyplot as plt
```

### 함수 정의

```python
def xy_to_XYZ(xy, Y=1.0):
    x, y = xy
    X = (x / y) * Y
    Z = ((1 - x - y) / y) * Y
    return np.array([X, Y, Z], dtype=np.float32)

def XYZ_to_xyY(XYZ):
    xyz_array = np.array(XYZ)
    X, Y, Z = xyz_array[..., 0], xyz_array[..., 1], xyz_array[..., 2]
    sum_XYZ = X + Y + Z
    
    x = np.where(sum_XYZ != 0, X / sum_XYZ, 0)
    y = np.where(sum_XYZ != 0, Y / sum_XYZ, 0)
    return np.stack([x, y, Y], axis=-1)

# def srgb_to_linear(srgb):
#     mask = srgb <= 0.04045
#     return np.where(mask, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

# def linear_to_srgb(linear):
#     mask = linear <= 0.0031308
#     return np.where(mask, linear * 12.92, 1.055 * np.power(linear, 1/2.4) - 0.055)

def srgb_to_linear(srgb):
    """ srgb[0, 1] to linear RGB[0, 1] """
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    mask = srgb <= 0.04045
    linear = np.empty_like(srgb)
    linear[mask] = srgb[mask] / 12.92
    linear[~mask] = np.power((srgb[~mask] + 0.055)/1.055, 2.4)
    return linear

def linear_to_srgb(linear):
    """ linear RGB[0, 1] to srgb[0, 1] """
    linear = np.clip(linear, 0.0, 1.0).astype(np.float32)
    mask = linear <= 0.0031308
    srgb = np.empty_like(linear)
    srgb[mask] = linear[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(linear[~mask], 1.0/2.4) - 0.055
    return srgb

primaries = { 'R': (0.640, 0.330),
              'G': (0.300, 0.600),
              'B': (0.150, 0.060),
              'W': (0.3127, 0.3290)
}
```

### 색좌표 변환 함수

```python
def get_RGB2XYZ_matrix(primaries, Y_w):
    XYZ_R = xy_to_XYZ(primaries['R'], Y=1.0)
    XYZ_G = xy_to_XYZ(primaries['G'], Y=1.0)
    XYZ_B = xy_to_XYZ(primaries['B'], Y=1.0)
    XYZ_W = xy_to_XYZ(primaries['W'], Y=Y_w)

    M = np.stack([XYZ_R, XYZ_G, XYZ_B], axis=1) # (3, 3)
    S = np.linalg.solve(M, XYZ_W)               # (3,)
    M_scaled = M * S[np.newaxis, :]             # (3,3)
    return M_scaled

def RGB_to_XYZ(RGB, primaries, Y_w=1.0):
    linear = srgb_to_linear(RGB)
    M_RGB2XYZ = get_RGB2XYZ_matrix(primaries, Y_w)
    return linear.dot(M_RGB2XYZ.T)

def XYZ_to_RGB(XYZ, primaries, Y_w=1.0):
    M_RGB2XYZ = get_RGB2XYZ_matrix(primaries, Y_w)
    M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
    linear = XYZ.dot(M_XYZ2RGB.T)
    return linear_to_srgb(linear)

def XYZ_to_Lab(XYZ, primaries):
    Xn, Yn, Zn = xy_to_XYZ(primaries['W'], Y=1.0)
    # Xn, Yn, Zn = 0.95047, 1.00000, 1.08883    # D65
    xr = XYZ[..., 0] / Xn
    yr = XYZ[..., 1] / Yn
    zr = XYZ[..., 2] / Zn

    delta = 6/29.
    fx = np.where(xr > delta**3, np.cbrt(xr), xr/(3*delta**2) + 4/29)
    fy = np.where(yr > delta**3, np.cbrt(yr), yr/(3*delta**2) + 4/29)
    fz = np.where(zr > delta**3, np.cbrt(zr), zr/(3*delta**2) + 4/29)

    L = (116 * fy - 16).astype(np.float32)
    a = (500 * (fx - fy)).astype(np.float32)
    b = (200 * (fy - fz)).astype(np.float32)
    return np.stack([L, a, b], axis=-1)

def XYZ_to_Luv(XYZ, primaries):
    Xn, Yn, Zn = xy_to_XYZ(primaries['W'], Y=1.0)
    # Xn, Yn, Zn = 0.95047, 1.00000, 1.08883    # D65
    
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    denom = X + 15*Y + 3*Z
    denom = np.where(denom == 0, 1e-20, denom)
    u_prime = 4 * X / denom
    v_prime = 9 * Y / denom
    
    denom_n = Xn + 15*Yn + 3*Zn
    un_prime = 4 * Xn / denom_n
    vn_prime = 9 * Yn / denom_n
    
    # L*
    yr, eps, kappa = Y / Yn, 0.008856, 903.3
    L = np.where(yr > eps, 116*np.cbrt(yr) - 16, yr * kappa)
    u = 13 * L * (u_prime - un_prime)
    v = 13 * L * (v_prime - vn_prime)
    return np.stack([L, u, v], axis=-1)
```

### 함수 테스트

```python
red   = np.r_[1., 0., 0.]
green = np.r_[0., 1., 0.]
blue  = np.r_[0., 0., 1.]
white = np.r_[1., 1., 1.]
gray  = np.r_[.5, .5, .5]
orange = np.r_[255., 128., 64.] / 255.

RGB = orange
XYZ = RGB_to_XYZ(RGB, primaries)
xyY = XYZ_to_xyY(XYZ)
Lab = XYZ_to_Lab(XYZ, primaries)
Luv = XYZ_to_Luv(XYZ, primaries)

print(f"RGB: {RGB}")
print(f"XYZ: {XYZ}")
print(f"xyY: {xyY}")
print(f"Lab: {Lab}")
print(f"Luv: {Luv}")

import skimage

RGB = orange
XYZ = skimage.color.rgb2xyz(RGB)
xyY = XYZ_to_xyY(XYZ)
Lab = skimage.color.xyz2lab(XYZ)
Luv = skimage.color.xyz2luv(XYZ)

print(f"RGB: {RGB}")
print(f"XYZ: {XYZ}")
print(f"xyY: {xyY}")
print(f"Lab: {Lab}")
print(f"Luv: {Luv}")
```

### 데이터 분석

```python
def load_image(img_path):
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float32(img)
    if img.ndim == 2:
        img = skimage.color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img)
    return img

def load_data(x_path, y_path=None, z_path=None, dimming=1.0):
    if y_path is None or z_path is None:
        y_path = x_path.replace("_X.csv", "_Y.csv")
        z_path = x_path.replace("_X.csv", "_Z.csv")
        
    X = np.loadtxt(x_path, delimiter=',', dtype=np.float32)
    Y = np.loadtxt(y_path, delimiter=',', dtype=np.float32)
    Z = np.loadtxt(z_path, delimiter=',', dtype=np.float32)

    X = np.clip(X, 0, None) / dimming
    Y = np.clip(Y, 0, None) / dimming
    Z = np.clip(Z, 0, None) / dimming
    return np.stack([X, Y, Z], axis=-1)

def show_image(img):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 3), sharey=True)
    for i in range(3):
        axes[i].imshow(img[..., i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.tight_layout()
    plt.show()

img_path = "E:\\data_2024\\CPD2303\\images\\t2_4_i.png"
X_path = "E:\\data_2024\\CPD2303\\data\\t2_4_i 120 1000 60_X.csv"
Y_path = "E:\\data_2024\\CPD2303\\data\\t2_4_i 120 1000 60_Y.csv"
Z_path = "E:\\data_2024\\CPD2303\\data\\t2_4_i 120 1000 60_Z.csv"

dimming = 1000
primaries = { 'R': (0.690, 0.309),
              'G': (0.209, 0.748),
              'B': (0.142, 0.042),
              'W': (0.303, 0.314)
}

RGB_original = load_image(img_path)
XYZ_original = RGB_to_XYZ(RGB_original, primaries)
Lab_original = XYZ_to_Lab(XYZ_original, primaries)
Luv_original = XYZ_to_Luv(XYZ_original, primaries)

XYZ_measured = load_data(X_path, dimming=dimming)
RGB_measured = XYZ_to_RGB(XYZ_measured, primaries)
Lab_measured = XYZ_to_Lab(XYZ_measured, primaries)
Luv_measured = XYZ_to_Luv(XYZ_measured, primaries)
```

### 비교

```python
show_image(XYZ_original)
show_image(XYZ_measured)

# RGB_original = skimage.color.xyz2rgb(XYZ_original)
# RGB_measured = skimage.color.xyz2rgb(XYZ_measured)

show_image(RGB_original)
show_image(RGB_measured)
```
