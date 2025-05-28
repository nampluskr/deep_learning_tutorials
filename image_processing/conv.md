### Test

```python
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
from skimage import img_as_float
```

### Conversion functions

```python
def rgb8bit_to_srgb(rgb8bit):
    arr = np.array(rgb8bit, dtype=np.float64)
    return arr / 255.0

def srgb_to_rgb8bit(srgb):
    arr = np.array(srgb, dtype=np.float64)
    arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype(np.uint8)

def srgb_to_linear(srgb):
    arr = np.array(srgb, dtype=np.float64)
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return np.clip(arr, 0.0, 1.0)

def linear_to_srgb(linear):
    arr = np.array(linear,np.float64)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    return np.clip(arr, 0.0, 1.0)

def apply_dimming(sRGB, dimming):
    """ sRGB[0, 1] -> linear -> dimming -> sRGB[0, 1] """
    linear = srgb_to_linear(sRGB)
    linear_dimmed = linear * dimming
    srgb_dimmed = linear_to_srgb(linear_dimmed).copy()
    return np.clip(srgb_dimmed, 0.0, 1.0)
```

### D65 Illuminat (ITU-R BT.709)

```python
# D65 illuminant (ITU-R BT.709 Matrix)
linear_to_xyz_matrix = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])
        
# XYZ to sRGB 변환 매트릭스 (역변환용)
xyz_to_linear_matrix = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
])

# sRGB -> Linear RGB -> XYZ (skimage.color.rgb2xyz)
def srgb_to_xyz(srgb):
    return srgb_to_linear(srgb).dot(linear_to_xyz_matrix.T)

# XYZ -> Linear RGB -> sRGB (skimage.color.xyz2rgb)
def xyz_to_srgb(xyz):
    linear = xyz.dot(xyz_to_linear_matrix.T)
    return linear_to_srgb(linear)

img = skimage.data.astronaut()      # 8bit RGB or RGB_uint8
img_srgb = rgb8bit_to_srgb(img)     # img_as_float(img)
img_srgb_dimmed = apply_dimming(img_srgb, dimming=1)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 4), ncols=3)
ax1.imshow(img)
ax2.imshow(img_srgb)
ax3.imshow(img_srgb_dimmed)
fig.tight_layout()
plt.show()
```

### etc

```python
def XYZ_to_Lxy(XYZ):
    XYZ = np.asfarray(XYZ)
    if XYZ.ndim == 1:
        XYZ_sum = np.sum(XYZ)
        if XYZ_sum == 0:
            return 0.0, 0.0
        x = XYZ[0] / XYZ_sum
        y = XYZ[1] / XYZ_sum
    else:
        XYZ_sum = np.sum(XYZ, axis=1)
        XYZ_sum = np.where(XYZ_sum == 0, np.finfo(float).eps, XYZ_sum)
        x = XYZ[:, 0] / XYZ_sum
        y = XYZ[:, 1] / XYZ_sum
    return x, y

def get_rgb_to_xyz_matrix(Rx, Ry, Gx, Gy, Bx, By, Wx, Wy):
    matrix = np.array([
        [Rx/Ry, Gx/Gy, Bx/By],
        [1.0, 1.0, 1.0],
        [(1 - Rx - Ry)/Ry, (1 - Gx - Gy)/Gy, (1 - Bx - By)/By]
    ])
    white_XYZ = np.array([Wx/Wy, 1.0, (1 - Wx - Wy)/Wy])
    scale = np.linalg.solve(matrix, white_XYZ)
    return matrix * scale

# linear_to_xyz_D65 = np.array([
#     [0.4124564, 0.3575761, 0.1804375],
#     [0.2126729, 0.7151522, 0.0721750],
#     [0.0193339, 0.1191920, 0.9503041]
# ])

Rx, Ry = 0.64, 0.33
Gx, Gy = 0.30, 0.60
Bx, By = 0.15, 0.06
Wx, Wy = 0.3127, 0.3290

get_rgb_to_xyz_matrix(Rx, Ry, Gx, Gy, Bx, By, Wx, Wy)
```

### Class

```python
class Color:
    def __init__(self, red=(0.64, 0.33), green=(0.30, 0.60), blue=(0.15, 0.06), 
                 white=(0.3127, 0.3290), max_luminace=1):
        self.Rx, self.Ry = red
        self.Gx, self.Gy = green
        self.Bx, self.By = blue
        self.Wx, self.Wy = white
        self.Linear_to_XYZ = self.get_rgb_to_xyz_matrix()
        self.XYZ_to_Linear = np.linalg.inv(self.Linear_to_XYZ)

    def get_RGB_to_XYZ_matrix(self):
        matrix = np.array([
            [self.Rx/self.Ry, self.Gx/self.Gy, self.Bx/self.By],
            [1.0, 1.0, 1.0],
            [(1-self.Rx-self.Ry)/self.Ry, (1-self.Gx-self.Gy)/Gy, (1-self.Bx-self.By)/self.By]
        ])
        white_XYZ = np.array([self.Wx/self.Wy, 1.0, (1-self.Wx-self.Wy)/self.Wy])
        scale = np.linalg.solve(matrix, white_XYZ)
        return matrix * scale

    def set_max_luminace(self, Lmax): self.Lmax = Lmax
    def linear_to_srgb(self, linear): pass
    def srgb_to_linear(self, srgb): pass
    def linear_to_xyz(self, linear): pass
    def xyz_to_linear(self, xyz): pass   
```
