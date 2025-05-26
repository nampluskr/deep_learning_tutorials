```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from pyDOE import lhs   # Latin Hypercube Sampling
```

## Problem: Damped Harmonic Oscillator

- https://github.com/benmoseley/harmonic-oscillator-pinn/blob/main/Harmonic%20oscillator%20PINN.ipynb

$$m\frac{d^2 u}{dt^2} + \mu\frac{du}{dt} + ku = 0$$

Initial conditions:
$$u(0) = 1,\quad \frac{du}{dt}(0) = 0$$

Under-damped state:
$$\delta < \omega_0,\quad \text{with}\quad \delta = \frac{\mu}{2m},\quad \omega_0 = \sqrt{\frac{k}{m}}$$

Exact solution:
$$u(t) = e^{-\delta t}(2A\cos(\phi + \omega t)),
\quad\text{with}\quad \omega = \sqrt{\omega_0^2 - \delta^2},
\quad\phi = \tan^{-1}\left(-\frac{\delta}{\omega_0}\right),
\quad A = \frac{1}{2\cos\phi}$$

### Training Points - Sampling

```python
def u_exact(t, m=1, mu=4, k=400):
    delta, omega0 = mu/m/2, np.sqrt(k/m)
    omega = np.sqrt(omega0**2 - delta**2)
    phi = np.arctan(-delta/omega0)
    A = 1/np.cos(phi)/2
    return 2*A * np.cos(phi + omega*t) * np.exp(-delta*t)

np.random.seed(42)
# Data points: (t_data, u_data)
n_data, noise_level = 20, 0.02
t_data = lhs(1, samples=n_data)
u_data = u_exact(t_data) + np.random.normal(0, noise_level, t_data.shape)

# Collocation points: t in (0, 1)
n_points = 101
t = lhs(1, samples=n_points)    # Latin Hypercube Sampling

# Initial Conditions: u(0) = 1, u'(0) = 0
t_ic = np.array([0.0])
u_ic = np.array([1.0])
du_ic = np.array([0.0])

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t_data, u_data, 'bo', label='Data points')
ax.plot(t, np.zeros_like(t), 'kx', label='Collocation points')
ax.plot(t_ic, np.zeros_like(t_ic), 'ro', label="IC: u(0) = 1, u'(0)=0")
ax.legend()
fig.tight_layout()
plt.show()
```

### Modeling

```python
class PINN(nn.Module):
    def __init__(self, mu=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 50),  nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
        )
        self.mu = nn.Parameter(torch.tensor(mu).float())

    def forward(self, x):
        return self.layers(x)

# dy/dx using autograd
def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

# u_tt + mu*u_t + k*u = 0
def loss_pde(model, t, m, k):
    t.requires_grad = True
    u = model(t)
    u_t = gradient(u, t)
    u_tt = gradient(u_t, t)
    residual = m*u_tt + model.mu*u_t + k*u
    return torch.mean(residual**2)

# u_t(0) = 0
def loss_du(model, t, du_true):
    t.requires_grad = True
    u = model(t)
    du = gradient(u, t)
    return torch.mean((du - du_true)**2)

# u(0) = 1
def loss_mse(model, t, u_true):
    u = model(t)
    return torch.mean((u - u_true)**2)

# np.ndarray to torch tensor
def tensor(x):
    return torch.tensor(x).float().view(-1, 1)
```

### Training

```python
# Hyperparameters
m, mu, k = 1, 4, 400
learning_rate = 1e-3
n_epochs = 20000

model = PINN(mu=1.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

losses = {}
for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()

    losses["data"] = loss_mse(model, tensor(t_data), tensor(u_data))
    losses["pde"] = loss_pde(model, tensor(t), k)
    losses["ic1"] = loss_du(model, tensor(t_ic), tensor(du_ic))
    losses["ic2"] = loss_mse(model, tensor(t_ic), tensor(u_ic))
    loss = losses["data"] + 1e-4 * losses["pde"] + 1e-2 * losses["ic1"] + losses["ic2"]

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] "
              f"losses: {loss.item():.3e} (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"data: {losses['data'].item():.3e} "
              f"pde: {losses['pde'].item():.3e} "
              f"ic1: {losses['ic1'].item():.3e} "
              f"ic2: {losses['ic2'].item():.3e} " 
              f"mu: {model.mu.item():.3f}")
```

```
[ 2000/20000] losses: 4.289e-02 (lr: 9.02e-04) data: 1.414e-02 pde: 2.827e+02 ic1: 1.792e-04 ic2: 4.850e-04 mu: 3.438
[ 4000/20000] losses: 6.714e-03 (lr: 8.15e-04) data: 2.262e-03 pde: 4.451e+01 ic1: 2.860e-05 ic2: 1.123e-06 mu: 4.267
[ 6000/20000] losses: 2.519e-03 (lr: 7.35e-04) data: 1.251e-03 pde: 1.265e+01 ic1: 6.900e-05 ic2: 1.938e-06 mu: 4.417
[ 8000/20000] losses: 2.318e-03 (lr: 6.63e-04) data: 7.283e-04 pde: 1.577e+01 ic1: 6.328e-04 ic2: 6.936e-06 mu: 4.246
[10000/20000] losses: 9.076e-04 (lr: 5.99e-04) data: 4.424e-04 pde: 4.650e+00 ic1: 1.227e-05 ic2: 2.789e-08 mu: 4.099
[12000/20000] losses: 6.956e-04 (lr: 5.40e-04) data: 3.521e-04 pde: 3.432e+00 ic1: 4.365e-06 ic2: 2.125e-07 mu: 4.019
[14000/20000] losses: 6.146e-04 (lr: 4.88e-04) data: 3.293e-04 pde: 2.853e+00 ic1: 1.272e-08 ic2: 9.867e-08 mu: 3.983
[16000/20000] losses: 5.517e-04 (lr: 4.40e-04) data: 3.147e-04 pde: 2.362e+00 ic1: 6.675e-06 ic2: 8.166e-07 mu: 3.967
[18000/20000] losses: 1.260e-03 (lr: 3.97e-04) data: 3.870e-04 pde: 8.634e+00 ic1: 6.017e-04 ic2: 3.136e-06 mu: 3.958
[20000/20000] losses: 4.662e-04 (lr: 3.58e-04) data: 3.089e-04 pde: 1.567e+00 ic1: 1.692e-06 ic2: 5.978e-07 mu: 3.950
```

### Evaluation

```python
t_test = np.linspace(0, 1, 101)
with torch.no_grad():
    model.eval()
    u_pred = model(tensor(t_test))

# Plotting the results
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(t_test, u_exact(t_test), 'kx', ms=5, label="Exact")
ax.plot(t_data, u_data, 'bo', label="Data", ms=5)
ax.plot(t_test, u_pred, 'r', lw=2, label="PINN")
ax.set(xlabel="t", ylabel="u(t)")
ax.grid(color="k", ls=":", lw=1)
ax.legend()
plt.show()
```
