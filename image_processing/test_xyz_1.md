```python
import numpy as np
import matplotlib.pyplot as plt
import skimage

img_path = "E:\\data_2024\\CPD2303\\images\\t2_4_i.png"
X_path = "E:\\data_2024\\CPD2303\\data\\t2_4_i 120 110 60_X.csv"
Y_path = "E:\\data_2024\\CPD2303\\data\\t2_4_i 120 110 60_Y.csv"
Z_path = "E:\\data_2024\\CPD2303\\data\\t2_4_i 120 110 60_Z.csv"

# img_path = img_path = "E:\\data_2024\\CT3\\images\\t2_4_d.png"
# X_path = 'E:\\data_2024\\CT3\\S_L_Optimization_1\\t2_4_d 120 183 70_X.csv'
# Y_path = 'E:\\data_2024\\CT3\\S_L_Optimization_1\\t2_4_d 120 183 70_Y.csv'
# Z_path = 'E:\\data_2024\\CT3\\S_L_Optimization_1\\t2_4_d 120 183 70_Z.csv'

config = {"Rx": 0.64, "Ry": 0.33, "Gx": 0.30, "Gy": 0.60, 
          "Bx": 0.15, "By": 0.06, "Wx": 0.3127, "Wy": 0.3290}
config["height"] = 1920
config["width"] = 1080
dimming = config["dimming"] = 110
```

```python
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

def apply_dimming(srgb, dimming):
    linear = srgb_to_linear(srgb)
    linRGB_dimmed = np.clip(linear*dimming, 0.0, 1.0)
    srgb = linear_to_srgb(linRGB_dimmed)
    return srgb

def get_RGB2XYZ(Rx=0.64, Ry=0.33, Gx=0.30, Gy=0.60, 
                Bx=0.15, By=0.06, Wx=0.3127, Wy=0.3290):
    red    = np.r_[Rx, Ry, 1 - Rx - Ry] / Ry
    green  = np.r_[Gx, Gy, 1 - Gx - Gy] / Gy
    blue   = np.r_[Bx, By, 1 - Bx - By] / By
    white  = np.r_[Wx, Wy, 1 - Wx - Wy] / Wy

    matrix = np.stack([red, green, blue], axis=-1).astype(np.float32)
    scale  = np.linalg.solve(matrix, white).astype(np.float32)
    return matrix * scale

def sRGB_to_XYZ(srgb):
    linear = srgb_to_linear(srgb)
    RGB2XYZ = get_RGB2XYZ()
    return linear.dot(RGB2XYZ.T)

def XYZ_to_sRGB(XYZ):
    RGB2XYZ = get_RGB2XYZ()
    XYZ2RGB = np.linalg.inv(RGB2XYZ)
    linear = XYZ.dot(XYZ2RGB.T)
    return linear_to_srgb(linear)
```

```python
def load_image(img_path):
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float32(img)
    if img.ndim == 2:
        img = skimage.color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = skimage.color.rgba2rgb(img)
    return img

def load_data(x_path, y_path=None, z_path=None, dimming=1):
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

img_RGB = load_image(img_path)                  # sRGB [0, 1]
# img_XYZ = skimage.color.rgb2xyz(img_RGB)
img_XYZ = sRGB_to_XYZ(img_RGB)                  # sRGB [0, 1] -> XYZ [0, 1]
data_XYZ = load_data(X_path, dimming=dimming)   # XYZ [0, 1]
data_RGB = XYZ_to_sRGB(data_XYZ)                # XYZ [0, 1] -> sRGB [0, 1]
```

```python
def show_image(img):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 3), sharey=True)
    for i in range(3):
        axes[i].imshow(img[..., i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.tight_layout()
    plt.show()

show_image(img_XYZ)
show_image(data_XYZ)

show_image(img_RGB)
show_image(data_RGB)
```

```python
diff =abs(img_RGB - data_RGB)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3), sharey=True)
ax1.imshow(img_RGB)
ax2.imshow(data_RGB)
ax3.imshow(diff)
fig.tight_layout()
plt.show()

show_image(diff)
```

```python
def plot_hist(ax, arr, bins=200, **kwargs):
    data = arr.flatten()
    ax.hist(data, bins=bins, color="gray", edgecolor="black", **kwargs)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xlim(0, 1)
    ax.grid()
    return ax

delta = abs(data_XYZ - img_XYZ)
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3), sharey=True)
ax1 = plot_hist(ax1, delta[..., 0])
ax2 = plot_hist(ax2, delta[..., 1])
ax3 = plot_hist(ax3, delta[..., 2])
fig.tight_layout()
plt.show()

show_image(delta)
```

```python
def XYZ_to_Lxy(XYZ):
    eps = 1e-6
    X = XYZ[..., 0]
    Y = XYZ[..., 1]
    Z = XYZ[..., 2]
    mask = Y >= eps
    sum_XYZ = X + Y + Z
    x = np.zeros_like(X)
    y = np.zeros_like(Y)
    x[mask] = X[mask] / sum_XYZ[mask]
    y[mask] = Y[mask] / sum_XYZ[mask]
    return np.stack([Y, x, y], axis=-1)

img_XYZ = sRGB_to_XYZ(img_RGB)
img_Lxy = XYZ_to_Lxy(img_XYZ)

show_image(img_Lxy)
```
