### 수정2
```
==================================================
RUN EXPERIMENT: GRID - BASELINE MODEL
==================================================

 > Train set: 264 images, Normal: 264, Anomaly: 0
 > Test set: 78 images, Normal: 21, Anomaly: 57

 > Total params.:     270,015,619
 > Trainable params.: 270,015,619

 > Start training...

 [  1/50] loss=0.654, mse=0.506, ssim=0.002 | (val) loss=0.585, mse=0.409, ssim=0.004 (5.8s)
 [  2/50] loss=0.643, mse=0.492, ssim=0.004 | (val) loss=0.583, mse=0.406, ssim=0.005 (5.1s)
 [  3/50] loss=0.637, mse=0.483, ssim=0.004 | (val) loss=0.574, mse=0.392, ssim=0.004 (4.9s)
 [  4/50] loss=0.624, mse=0.465, ssim=0.005 | (val) loss=0.569, mse=0.384, ssim=-0.000 (2.3s)
 [  5/50] loss=0.583, mse=0.419, ssim=0.036 | (val) loss=0.505, mse=0.320, ssim=0.061 (4.9s)
 [  6/50] loss=0.481, mse=0.318, ssim=0.139 | (val) loss=0.443, mse=0.265, ssim=0.142 (4.9s)
 [  7/50] loss=0.397, mse=0.240, ssim=0.237 | (val) loss=0.397, mse=0.226, ssim=0.204 (4.9s)
 [  8/50] loss=0.353, mse=0.204, ssim=0.300 | (val) loss=0.371, mse=0.205, ssim=0.241 (4.9s)
 [  9/50] loss=0.323, mse=0.180, ssim=0.343 | (val) loss=0.352, mse=0.188, ssim=0.267 (4.9s)
 [ 10/50] loss=0.302, mse=0.164, ssim=0.375 | (val) loss=0.338, mse=0.178, ssim=0.287 (4.8s)
 > Image-level: auroc=0.892, aupr=0.954, th=2.044, acc=0.897, prec=0.877, recall=1.000, f1=0.934 | tp=57, tn=13, fp= 8, fn= 0 (f1)
 > Image-level: auroc=0.892, aupr=0.954, th=2.673, acc=0.821, prec=0.939, recall=0.807, f1=0.868 | tp=46, tn=18, fp= 3, fn=11 (roc)
 > Image-level: auroc=0.892, aupr=0.954, th=3.181, acc=0.628, prec=0.937, recall=0.526, f1=0.674 | tp=30, tn=19, fp= 2, fn=27 (percentile)
 > Pixel-level: auroc=0.698, aupr=0.079, iou=0.044, dice=0.078, pro=0.254

 [ 11/50] loss=0.285, mse=0.150, ssim=0.401 | (val) loss=0.329, mse=0.170, ssim=0.302 (5.0s)
 [ 12/50] loss=0.273, mse=0.142, ssim=0.423 | (val) loss=0.320, mse=0.165, ssim=0.319 (4.9s)
 [ 13/50] loss=0.261, mse=0.134, ssim=0.443 | (val) loss=0.315, mse=0.161, ssim=0.323 (4.9s)
 [ 14/50] loss=0.251, mse=0.127, ssim=0.460 | (val) loss=0.309, mse=0.157, ssim=0.335 (4.9s)
 [ 15/50] loss=0.242, mse=0.120, ssim=0.475 | (val) loss=0.302, mse=0.153, ssim=0.348 (4.8s)
 [ 16/50] loss=0.237, mse=0.117, ssim=0.484 | (val) loss=0.299, mse=0.150, ssim=0.353 (2.3s)
 [ 17/50] loss=0.229, mse=0.112, ssim=0.497 | (val) loss=0.296, mse=0.149, ssim=0.359 (4.9s)
 [ 18/50] loss=0.221, mse=0.107, ssim=0.512 | (val) loss=0.299, mse=0.150, ssim=0.353 (4.9s)
 [ 19/50] loss=0.217, mse=0.104, ssim=0.520 | (val) loss=0.290, mse=0.145, ssim=0.369 (4.9s)
 [ 20/50] loss=0.213, mse=0.102, ssim=0.527 | (val) loss=0.288, mse=0.144, ssim=0.374 (4.9s)
 > Image-level: auroc=0.931, aupr=0.973, th=1.980, acc=0.885, prec=0.875, recall=0.982, f1=0.926 | tp=56, tn=13, fp= 8, fn= 1 (f1)
 > Image-level: auroc=0.931, aupr=0.973, th=2.431, acc=0.885, prec=0.944, recall=0.895, f1=0.919 | tp=51, tn=18, fp= 3, fn= 6 (roc)
 > Image-level: auroc=0.931, aupr=0.973, th=2.913, acc=0.744, prec=0.951, recall=0.684, f1=0.796 | tp=39, tn=19, fp= 2, fn=18 (percentile)
 > Pixel-level: auroc=0.729, aupr=0.101, iou=0.052, dice=0.090, pro=0.277

 [ 21/50] loss=0.208, mse=0.099, ssim=0.536 | (val) loss=0.286, mse=0.142, ssim=0.377 (2.3s)
 [ 22/50] loss=0.202, mse=0.095, ssim=0.547 | (val) loss=0.284, mse=0.140, ssim=0.381 (4.9s)
 [ 23/50] loss=0.198, mse=0.093, ssim=0.555 | (val) loss=0.282, mse=0.139, ssim=0.385 (4.9s)
 [ 24/50] loss=0.195, mse=0.091, ssim=0.561 | (val) loss=0.282, mse=0.139, ssim=0.385 (4.9s)
 [ 25/50] loss=0.190, mse=0.087, ssim=0.571 | (val) loss=0.280, mse=0.138, ssim=0.390 (4.9s)
 [ 26/50] loss=0.187, mse=0.086, ssim=0.577 | (val) loss=0.279, mse=0.137, ssim=0.391 (5.0s)
 [ 27/50] loss=0.184, mse=0.084, ssim=0.583 | (val) loss=0.277, mse=0.136, ssim=0.393 (4.8s)
 [ 28/50] loss=0.182, mse=0.083, ssim=0.587 | (val) loss=0.274, mse=0.135, ssim=0.399 (2.2s)
 [ 29/50] loss=0.178, mse=0.081, ssim=0.594 | (val) loss=0.275, mse=0.135, ssim=0.399 (4.9s)
 [ 30/50] loss=0.177, mse=0.080, ssim=0.597 | (val) loss=0.273, mse=0.134, ssim=0.402 (4.9s)
 > Image-level: auroc=0.948, aupr=0.980, th=1.782, acc=0.897, prec=0.877, recall=1.000, f1=0.934 | tp=57, tn=13, fp= 8, fn= 0 (f1)
 > Image-level: auroc=0.948, aupr=0.980, th=2.308, acc=0.897, prec=0.962, recall=0.895, f1=0.927 | tp=51, tn=19, fp= 2, fn= 6 (roc)
 > Image-level: auroc=0.948, aupr=0.980, th=2.828, acc=0.769, prec=0.953, recall=0.719, f1=0.820 | tp=41, tn=19, fp= 2, fn=16 (percentile)
 > Pixel-level: auroc=0.742, aupr=0.108, iou=0.054, dice=0.094, pro=0.286

 [ 31/50] loss=0.174, mse=0.078, ssim=0.603 | (val) loss=0.269, mse=0.132, ssim=0.409 (4.9s)
 [ 32/50] loss=0.171, mse=0.077, ssim=0.608 | (val) loss=0.270, mse=0.132, ssim=0.409 (4.9s)
 [ 33/50] loss=0.170, mse=0.076, ssim=0.611 | (val) loss=0.271, mse=0.132, ssim=0.407 (2.3s)
 [ 34/50] loss=0.169, mse=0.075, ssim=0.613 | (val) loss=0.274, mse=0.133, ssim=0.397 (4.9s)
 [ 35/50] loss=0.164, mse=0.073, ssim=0.622 | (val) loss=0.268, mse=0.130, ssim=0.410 (4.9s)
 [ 36/50] loss=0.161, mse=0.071, ssim=0.628 | (val) loss=0.269, mse=0.130, ssim=0.408 (4.9s)
 [ 37/50] loss=0.160, mse=0.070, ssim=0.632 | (val) loss=0.266, mse=0.130, ssim=0.415 (5.0s)
 [ 38/50] loss=0.157, mse=0.069, ssim=0.636 | (val) loss=0.265, mse=0.128, ssim=0.417 (4.9s)
 [ 39/50] loss=0.155, mse=0.068, ssim=0.641 | (val) loss=0.267, mse=0.129, ssim=0.412 (2.3s)
 [ 40/50] loss=0.154, mse=0.067, ssim=0.644 | (val) loss=0.265, mse=0.128, ssim=0.415 (4.9s)
 > Image-level: auroc=0.949, aupr=0.980, th=1.903, acc=0.910, prec=0.903, recall=0.982, f1=0.941 | tp=56, tn=15, fp= 6, fn= 1 (f1)
 > Image-level: auroc=0.949, aupr=0.980, th=2.182, acc=0.910, prec=0.946, recall=0.930, f1=0.938 | tp=53, tn=18, fp= 3, fn= 4 (roc)
 > Image-level: auroc=0.949, aupr=0.980, th=2.724, acc=0.769, prec=0.953, recall=0.719, f1=0.820 | tp=41, tn=19, fp= 2, fn=16 (percentile)
 > Pixel-level: auroc=0.735, aupr=0.113, iou=0.054, dice=0.094, pro=0.277

 [ 41/50] loss=0.153, mse=0.066, ssim=0.646 | (val) loss=0.265, mse=0.128, ssim=0.416 (5.3s)
 [ 42/50] loss=0.151, mse=0.065, ssim=0.650 | (val) loss=0.263, mse=0.127, ssim=0.419 (5.2s)
 [ 43/50] loss=0.149, mse=0.065, ssim=0.653 | (val) loss=0.263, mse=0.127, ssim=0.420 (5.1s)
 [ 44/50] loss=0.148, mse=0.064, ssim=0.654 | (val) loss=0.265, mse=0.128, ssim=0.416 (2.4s)
 [ 45/50] loss=0.148, mse=0.064, ssim=0.656 | (val) loss=0.264, mse=0.127, ssim=0.418 (5.1s)
 [ 46/50] loss=0.146, mse=0.063, ssim=0.660 | (val) loss=0.263, mse=0.127, ssim=0.419 (5.1s)
 [ 47/50] loss=0.145, mse=0.062, ssim=0.661 | (val) loss=0.261, mse=0.126, ssim=0.422 (5.1s)
 [ 48/50] loss=0.143, mse=0.061, ssim=0.667 | (val) loss=0.262, mse=0.126, ssim=0.421 (5.1s)
 [ 49/50] loss=0.141, mse=0.060, ssim=0.670 | (val) loss=0.261, mse=0.125, ssim=0.423 (5.0s)
 [ 50/50] loss=0.140, mse=0.060, ssim=0.672 | (val) loss=0.262, mse=0.126, ssim=0.421 (2.5s)
 > Image-level: auroc=0.948, aupr=0.982, th=2.208, acc=0.897, prec=0.962, recall=0.895, f1=0.927 | tp=51, tn=19, fp= 2, fn= 6 (f1)
 > Image-level: auroc=0.948, aupr=0.982, th=2.286, acc=0.897, prec=0.980, recall=0.877, f1=0.926 | tp=50, tn=20, fp= 1, fn= 7 (roc)
 > Image-level: auroc=0.948, aupr=0.982, th=2.286, acc=0.885, prec=0.962, recall=0.877, f1=0.917 | tp=50, tn=19, fp= 2, fn= 7 (percentile)
 > Pixel-level: auroc=0.740, aupr=0.118, iou=0.055, dice=0.095, pro=0.277


 > Training finished...

 > Model weights saved to: ./results/grid/model_grid_baseline_epochs-50.pth
```

### 수정1
```
==================================================
RUN EXPERIMENT: GRID - BASELINE MODEL
==================================================

 > Train set: 264 images, Normal: 264, Anomaly: 0
 > Test set: 78 images, Normal: 21, Anomaly: 57

 > Start training...

 [  1/50] loss=0.752, mse=0.507, ssim=0.003 | (val) loss=0.702, mse=0.411, ssim=0.007 (10.8s)
 [  2/50] loss=0.744, mse=0.495, ssim=0.008 | (val) loss=0.699, mse=0.410, ssim=0.012 (3.0s)
 [  3/50] loss=0.742, mse=0.493, ssim=0.010 | (val) loss=0.698, mse=0.409, ssim=0.013 (4.8s)
 [  4/50] loss=0.739, mse=0.490, ssim=0.012 | (val) loss=0.694, mse=0.402, ssim=0.015 (4.8s)
 [  5/50] loss=0.714, mse=0.457, ssim=0.030 | (val) loss=0.646, mse=0.356, ssim=0.063 (5.0s)
 [  6/50] loss=0.614, mse=0.346, ssim=0.118 | (val) loss=0.567, mse=0.271, ssim=0.137 (5.0s)
 [  7/50] loss=0.510, mse=0.251, ssim=0.231 | (val) loss=0.515, mse=0.232, ssim=0.203 (4.9s)
 [  8/50] loss=0.459, mse=0.215, ssim=0.296 | (val) loss=0.479, mse=0.205, ssim=0.247 (2.8s)
 [  9/50] loss=0.420, mse=0.186, ssim=0.347 | (val) loss=0.459, mse=0.190, ssim=0.271 (4.9s)
 [ 10/50] loss=0.396, mse=0.170, ssim=0.379 | (val) loss=0.442, mse=0.180, ssim=0.296 (4.7s)
 > img_auroc=0.871, img_aupr=0.942, pix_auroc=0.706, pix_aupr=0.075

 [ 11/50] loss=0.375, mse=0.156, ssim=0.406 | (val) loss=0.431, mse=0.171, ssim=0.310 (5.3s)
 [ 12/50] loss=0.359, mse=0.147, ssim=0.429 | (val) loss=0.424, mse=0.167, ssim=0.320 (4.9s)
 [ 13/50] loss=0.344, mse=0.138, ssim=0.449 | (val) loss=0.414, mse=0.162, ssim=0.335 (4.7s)
 [ 14/50] loss=0.333, mse=0.132, ssim=0.465 | (val) loss=0.409, mse=0.159, ssim=0.341 (2.8s)
 [ 15/50] loss=0.322, mse=0.125, ssim=0.481 | (val) loss=0.401, mse=0.154, ssim=0.353 (4.7s)
 [ 16/50] loss=0.316, mse=0.121, ssim=0.489 | (val) loss=0.395, mse=0.151, ssim=0.360 (4.7s)
 [ 17/50] loss=0.305, mse=0.116, ssim=0.505 | (val) loss=0.392, mse=0.150, ssim=0.366 (4.7s)
 [ 18/50] loss=0.296, mse=0.111, ssim=0.519 | (val) loss=0.391, mse=0.148, ssim=0.367 (4.8s)
 [ 19/50] loss=0.291, mse=0.108, ssim=0.526 | (val) loss=0.386, mse=0.146, ssim=0.374 (4.7s)
 [ 20/50] loss=0.285, mse=0.105, ssim=0.535 | (val) loss=0.382, mse=0.144, ssim=0.379 (4.7s)
 > img_auroc=0.921, img_aupr=0.961, pix_auroc=0.724, pix_aupr=0.098

 [ 21/50] loss=0.279, mse=0.102, ssim=0.544 | (val) loss=0.380, mse=0.143, ssim=0.384 (4.9s)
 [ 22/50] loss=0.273, mse=0.098, ssim=0.553 | (val) loss=0.377, mse=0.141, ssim=0.387 (4.7s)
 [ 23/50] loss=0.270, mse=0.097, ssim=0.558 | (val) loss=0.377, mse=0.141, ssim=0.388 (4.8s)
 [ 24/50] loss=0.264, mse=0.095, ssim=0.566 | (val) loss=0.374, mse=0.139, ssim=0.391 (4.9s)
 [ 25/50] loss=0.257, mse=0.091, ssim=0.577 | (val) loss=0.370, mse=0.138, ssim=0.397 (4.8s)
 [ 26/50] loss=0.254, mse=0.090, ssim=0.582 | (val) loss=0.367, mse=0.136, ssim=0.402 (2.9s)
 [ 27/50] loss=0.249, mse=0.087, ssim=0.589 | (val) loss=0.367, mse=0.137, ssim=0.402 (4.8s)
 [ 28/50] loss=0.248, mse=0.087, ssim=0.591 | (val) loss=0.366, mse=0.135, ssim=0.404 (4.9s)
 [ 29/50] loss=0.242, mse=0.084, ssim=0.599 | (val) loss=0.366, mse=0.135, ssim=0.404 (4.8s)
 [ 30/50] loss=0.240, mse=0.083, ssim=0.602 | (val) loss=0.363, mse=0.134, ssim=0.408 (4.8s)
 > img_auroc=0.941, img_aupr=0.977, pix_auroc=0.742, pix_aupr=0.105

 [ 31/50] loss=0.237, mse=0.082, ssim=0.608 | (val) loss=0.360, mse=0.133, ssim=0.412 (4.8s)
 [ 32/50] loss=0.234, mse=0.080, ssim=0.613 | (val) loss=0.359, mse=0.133, ssim=0.415 (2.7s)
 [ 33/50] loss=0.232, mse=0.079, ssim=0.616 | (val) loss=0.359, mse=0.131, ssim=0.413 (4.7s)
 [ 34/50] loss=0.228, mse=0.077, ssim=0.622 | (val) loss=0.359, mse=0.131, ssim=0.413 (4.8s)
 [ 35/50] loss=0.223, mse=0.075, ssim=0.630 | (val) loss=0.356, mse=0.130, ssim=0.418 (4.8s)
 [ 36/50] loss=0.220, mse=0.074, ssim=0.634 | (val) loss=0.353, mse=0.129, ssim=0.423 (4.7s)
 [ 37/50] loss=0.218, mse=0.073, ssim=0.638 | (val) loss=0.358, mse=0.131, ssim=0.416 (4.8s)
 [ 38/50] loss=0.215, mse=0.072, ssim=0.642 | (val) loss=0.353, mse=0.129, ssim=0.423 (2.6s)
 [ 39/50] loss=0.214, mse=0.072, ssim=0.643 | (val) loss=0.355, mse=0.129, ssim=0.419 (4.8s)
 [ 40/50] loss=0.211, mse=0.070, ssim=0.648 | (val) loss=0.353, mse=0.128, ssim=0.423 (4.6s)
 > img_auroc=0.933, img_aupr=0.973, pix_auroc=0.735, pix_aupr=0.112

 [ 41/50] loss=0.208, mse=0.069, ssim=0.653 | (val) loss=0.354, mse=0.128, ssim=0.420 (4.9s)
 [ 42/50] loss=0.206, mse=0.068, ssim=0.656 | (val) loss=0.356, mse=0.129, ssim=0.417 (4.8s)
 [ 43/50] loss=0.205, mse=0.068, ssim=0.658 | (val) loss=0.354, mse=0.128, ssim=0.420 (4.7s)
 [ 44/50] loss=0.203, mse=0.067, ssim=0.660 | (val) loss=0.352, mse=0.128, ssim=0.424 (2.3s)
 [ 45/50] loss=0.202, mse=0.067, ssim=0.663 | (val) loss=0.351, mse=0.127, ssim=0.424 (4.8s)
 [ 46/50] loss=0.200, mse=0.066, ssim=0.666 | (val) loss=0.349, mse=0.126, ssim=0.429 (4.7s)
 [ 47/50] loss=0.200, mse=0.065, ssim=0.666 | (val) loss=0.350, mse=0.126, ssim=0.426 (4.8s)
 [ 48/50] loss=0.196, mse=0.064, ssim=0.672 | (val) loss=0.351, mse=0.126, ssim=0.425 (4.7s)
 [ 49/50] loss=0.193, mse=0.063, ssim=0.676 | (val) loss=0.350, mse=0.126, ssim=0.425 (4.8s)
 [ 50/50] loss=0.193, mse=0.063, ssim=0.677 | (val) loss=0.350, mse=0.126, ssim=0.426 (4.6s)
 > img_auroc=0.942, img_aupr=0.979, pix_auroc=0.738, pix_aupr=0.113


 > Training finished...

 > Model weights saved to: ./results/grid/model_grid_baseline_epochs-50.pth
```


### 처음
```
*** RUN EXPERIMENT: AUTOENCODER - TILE
 > Train set: 230 images, Normal: 230, Anomaly: 0
 > Test set: 117 images, Normal: 33, Anomaly: 84
Epoch [1/20] loss=0.3165, ssim=0.0002 | (val) auroc=0.8377, aupr=0.9266
Epoch [2/20] loss=0.2927, ssim=0.0011 | (val) auroc=0.8351, aupr=0.9219
Epoch [3/20] loss=0.2876, ssim=0.0014 | (val) auroc=0.8478, aupr=0.9250
Epoch [4/20] loss=0.2858, ssim=0.0017 | (val) auroc=0.8510, aupr=0.9254
Epoch [5/20] loss=0.2844, ssim=0.0017 | (val) auroc=0.8496, aupr=0.9233
Epoch [6/20] loss=0.2830, ssim=0.0019 | (val) auroc=0.8611, aupr=0.9279
Epoch [7/20] loss=0.2810, ssim=0.0022 | (val) auroc=0.8586, aupr=0.9281
Epoch [8/20] loss=0.2771, ssim=0.0027 | (val) auroc=0.8571, aupr=0.9322
Epoch [9/20] loss=0.2741, ssim=0.0035 | (val) auroc=0.8600, aupr=0.9367
Epoch [10/20] loss=0.2690, ssim=0.0049 | (val) auroc=0.8380, aupr=0.9279
Epoch [11/20] loss=0.2598, ssim=0.0090 | (val) auroc=0.8431, aupr=0.9342
Epoch [12/20] loss=0.2494, ssim=0.0153 | (val) auroc=0.8647, aupr=0.9410
Epoch [13/20] loss=0.2405, ssim=0.0224 | (val) auroc=0.8369, aupr=0.9305
Epoch [14/20] loss=0.2286, ssim=0.0351 | (val) auroc=0.8546, aupr=0.9389
Epoch [15/20] loss=0.2122, ssim=0.0556 | (val) auroc=0.8496, aupr=0.9308
Epoch [16/20] loss=0.2014, ssim=0.0750 | (val) auroc=0.8582, aupr=0.9371
Epoch [17/20] loss=0.1875, ssim=0.1022 | (val) auroc=0.8359, aupr=0.9327
Epoch [18/20] loss=0.1775, ssim=0.1234 | (val) auroc=0.8377, aupr=0.9313
Epoch [19/20] loss=0.1657, ssim=0.1513 | (val) auroc=0.8341, aupr=0.9323
Epoch [20/20] loss=0.1565, ssim=0.1769 | (val) auroc=0.8196, aupr=0.9267

...Training finished.
            method  threshold     auroc    aupr  cohens_d  accuracy  precision  ...        f1       mcc     kappa  tn  fp  fn  tp
0              roc   2.859824  0.819625  0.9267  1.190706  0.777778   0.953125  ...  0.824324  0.574295  0.536563  30   3  23  61
1               f1   2.578285  0.819625  0.9267  1.190706  0.794872   0.894737  ...  0.850000  0.534881  0.528226  25   8  16  68
2    percentile_95   3.216363  0.819625  0.9267  1.190706  0.709402   0.962963  ...  0.753623  0.504089  0.437659  31   2  32  52
3    percentile_97   3.569411  0.819625  0.9267  1.190706  0.692308   0.980000  ...  0.731343  0.503064  0.421270  32   1  35  49
4    percentile_99   4.785182  0.819625  0.9267  1.190706  0.452991   0.954545  ...  0.396226  0.253016  0.139904  32   1  63  21
5          fbeta_2   1.483648  0.819625  0.9267  1.190706  0.735043   0.730435  ...  0.844221  0.210401  0.084784   2  31   0  84
6        fbeta_1.5   1.483648  0.819625  0.9267  1.190706  0.735043   0.730435  ...  0.844221  0.210401  0.084784   2  31   0  84
7          fbeta_1   2.590388  0.819625  0.9267  1.190706  0.794872   0.894737  ...  0.850000  0.534881  0.528226  25   8  16  68
8   cost_sensitive   1.483648  0.819625  0.9267  1.190706  0.735043   0.730435  ...  0.844221  0.210401  0.084784   2  31   0  84
9           3sigma   7.585437  0.819625  0.9267  1.190706  0.282051   0.000000  ...  0.000000  0.000000  0.000000  33   0  84   0
10          2sigma   6.190380  0.819625  0.9267  1.190706  0.324786   1.000000  ...  0.112360  0.132432  0.034472  33   0  79   5
11          1sigma   4.795322  0.819625  0.9267  1.190706  0.444444   0.952381  ...  0.380952  0.243658  0.131552  32   1  64  20
12       fixed_fpr        inf  0.819625  0.9267  1.190706  0.282051   0.000000  ...  0.000000  0.000000  0.000000  33   0  84   0

[13 rows x 15 columns]

*** RUN EXPERIMENT: AUTOENCODER - GRID
 > Train set: 264 images, Normal: 264, Anomaly: 0
 > Test set: 78 images, Normal: 21, Anomaly: 57
Epoch [1/20] loss=0.4886, ssim=0.0003 | (val) auroc=0.7251, aupr=0.8931
Epoch [2/20] loss=0.4769, ssim=-0.0009 | (val) auroc=0.7193, aupr=0.8872
Epoch [3/20] loss=0.4733, ssim=-0.0013 | (val) auroc=0.7352, aupr=0.8923
Epoch [4/20] loss=0.4679, ssim=-0.0011 | (val) auroc=0.7068, aupr=0.8711
Epoch [5/20] loss=0.4545, ssim=0.0023 | (val) auroc=0.8304, aupr=0.9211
Epoch [6/20] loss=0.3750, ssim=0.0662 | (val) auroc=0.7794, aupr=0.9016
Epoch [7/20] loss=0.2931, ssim=0.1511 | (val) auroc=0.7886, aupr=0.9038
Epoch [8/20] loss=0.2311, ssim=0.2295 | (val) auroc=0.8814, aupr=0.9489
Epoch [9/20] loss=0.1942, ssim=0.2860 | (val) auroc=0.8764, aupr=0.9470
Epoch [10/20] loss=0.1745, ssim=0.3218 | (val) auroc=0.8897, aupr=0.9548
Epoch [11/20] loss=0.1578, ssim=0.3532 | (val) auroc=0.8897, aupr=0.9539
Epoch [12/20] loss=0.1473, ssim=0.3743 | (val) auroc=0.9006, aupr=0.9560
Epoch [13/20] loss=0.1358, ssim=0.3989 | (val) auroc=0.8989, aupr=0.9592
Epoch [14/20] loss=0.1302, ssim=0.4135 | (val) auroc=0.9398, aupr=0.9776
Epoch [15/20] loss=0.1210, ssim=0.4353 | (val) auroc=0.9390, aupr=0.9750
Epoch [16/20] loss=0.1139, ssim=0.4519 | (val) auroc=0.9165, aupr=0.9633
Epoch [17/20] loss=0.1109, ssim=0.4610 | (val) auroc=0.9398, aupr=0.9752
Epoch [18/20] loss=0.1089, ssim=0.4705 | (val) auroc=0.9106, aupr=0.9577
Epoch [19/20] loss=0.1052, ssim=0.4787 | (val) auroc=0.9357, aupr=0.9717
Epoch [20/20] loss=0.0998, ssim=0.4912 | (val) auroc=0.9373, aupr=0.9726

...Training finished.
            method  threshold     auroc      aupr  cohens_d  accuracy  precision  ...        f1       mcc     kappa  tn  fp  fn  tp
0              roc   2.446214  0.937343  0.972592  1.853106  0.910256   0.962963  ...  0.936937  0.785216  0.781775  19   2   5  52
1               f1   2.148792  0.937343  0.972592  1.853106  0.923077   0.932203  ...  0.948276  0.800258  0.798450  17   4   2  55
2    percentile_95   3.256119  0.937343  0.972592  1.853106  0.628205   0.937500  ...  0.674157  0.388731  0.313297  19   2  27  30
3    percentile_97   3.278074  0.937343  0.972592  1.853106  0.628205   0.966667  ...  0.666667  0.420447  0.327986  20   1  28  29
4    percentile_99   3.300029  0.937343  0.972592  1.853106  0.628205   0.966667  ...  0.666667  0.420447  0.327986  20   1  28  29
5          fbeta_2   2.112949  0.937343  0.972592  1.853106  0.923077   0.918033  ...  0.949153  0.799726  0.792000  16   5   1  56
6        fbeta_1.5   2.112949  0.937343  0.972592  1.853106  0.923077   0.918033  ...  0.949153  0.799726  0.792000  16   5   1  56
7          fbeta_1   2.112949  0.937343  0.972592  1.853106  0.923077   0.918033  ...  0.949153  0.799726  0.792000  16   5   1  56
8   cost_sensitive   2.112949  0.937343  0.972592  1.853106  0.923077   0.918033  ...  0.949153  0.799726  0.792000  16   5   1  56
9           3sigma   6.287324  0.937343  0.972592  1.853106  0.282051   1.000000  ...  0.034483  0.069171  0.009524  21   0  56   1
10          2sigma   5.193396  0.937343  0.972592  1.853106  0.282051   1.000000  ...  0.034483  0.069171  0.009524  21   0  56   1
11          1sigma   4.099467  0.937343  0.972592  1.853106  0.448718   1.000000  ...  0.394366  0.283887  0.149163  21   0  43  14
12       fixed_fpr        inf  0.937343  0.972592  1.853106  0.269231   0.000000  ...  0.000000  0.000000  0.000000  21   0  57   0

[13 rows x 15 columns]
```