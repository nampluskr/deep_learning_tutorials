```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from pyDOE import lhs   # Latin Hypercube Sampling
```

## Problem: Damped Harmonic Oscillator

$$m\frac{d^2 u}{dt^2} + \mu\frac{du}{dt} + ku = 0,\quad u(0) = 1,\quad \frac{du}{dt}(0) = 0,\quad t\in[0, 1]$$

### Modeling

```python
def u_exact(t, d, w0):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    return 2*A * np.cos(phi + w*t) * np.exp(-d*t)

def gradient(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

def loss_pde(model, t, mu, k):
    t.requires_grad = True
    u = model(t)
    u_t = gradient(u, t)
    u_tt = gradient(u_t, t)
    return torch.mean((u_tt + mu*u_t + k*u)**2)

def loss_ic1(model, t, du_ic):
    t.requires_grad = True
    u = model(t)
    du = gradient(u, t)
    return torch.mean((du - du_ic)**2)

def loss_ic2(model, t, u_ic):
    u = model(t)
    return torch.mean((u - u_ic)**2)
```

### Training

```python
# u_tt + mu*u_t + k*u = 0
d, w0 = 2, 20
mu, k = 2*d, w0**2

# Hyperparameters
n_points = 101
learning_rate = 1e-3
n_epochs = 10000

# Collocation points: t in (0, 1)
torch.manual_seed(0)
t = lhs(1, samples=n_points)    # Latin Hypercube Sampling
t = torch.tensor(t).float().view(-1, 1)

# Initial Conditions: u(0) = 1, u'(0) = 0
t_ic = torch.tensor(0).float().view(-1, 1)
u_ic = torch.tensor(1).float().view(-1, 1)
du_ic = torch.tensor(0).float().view(-1, 1)

model = nn.Sequential(
            nn.Linear(1, 50),  nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1),
        )
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)

losses = {}
for epoch in range(1, n_epochs + 1):
    losses["pde"] = loss_pde(model, t, mu, k)
    losses["ic1"] = loss_ic1(model, t_ic, du_ic)
    losses["ic2"] = loss_ic2(model, t_ic, u_ic)
    loss = 1e-4 * losses["pde"] + 1e-2 * losses["ic1"] + losses["ic2"]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:5d}/{n_epochs}] "
              f"losses: {loss.item():.3e} (lr: {scheduler.get_last_lr()[0]:.2e}) "
              f"pde: {losses['pde'].item():.3e} "
              f"ic1: {losses['ic1'].item():.3e} "
              f"ic2: {losses['ic2'].item():.3e}")
```

```
[ 1000/10000] losses: 1.678e-02 (lr: 5.69e-04) pde: 1.614e+02 ic1: 1.738e-07 ic2: 6.488e-04
[ 2000/10000] losses: 4.727e-03 (lr: 5.40e-04) pde: 4.570e+01 ic1: 9.891e-06 ic2: 1.571e-04
[ 3000/10000] losses: 1.748e-03 (lr: 5.13e-04) pde: 1.708e+01 ic1: 8.982e-07 ic2: 4.002e-05
[ 4000/10000] losses: 7.435e-04 (lr: 4.88e-04) pde: 7.356e+00 ic1: 2.096e-06 ic2: 7.912e-06
[ 5000/10000] losses: 3.076e-04 (lr: 4.63e-04) pde: 3.060e+00 ic1: 2.585e-07 ic2: 1.584e-06
[ 6000/10000] losses: 1.986e-04 (lr: 4.40e-04) pde: 1.978e+00 ic1: 2.496e-07 ic2: 8.050e-07
[ 7000/10000] losses: 5.705e-04 (lr: 4.18e-04) pde: 5.407e+00 ic1: 2.910e-03 ic2: 6.633e-07
[ 8000/10000] losses: 1.299e-04 (lr: 3.97e-04) pde: 1.295e+00 ic1: 1.758e-08 ic2: 3.646e-07
[ 9000/10000] losses: 1.137e-04 (lr: 3.77e-04) pde: 1.134e+00 ic1: 1.997e-07 ic2: 3.016e-07
[10000/10000] losses: 1.015e-04 (lr: 3.58e-04) pde: 1.013e+00 ic1: 2.183e-10 ic2: 2.208e-07
```


### Evaluation

```python
t_test = np.linspace(0, 1, 101)
with torch.no_grad():
    t = torch.tensor(t_test).float().view(-1, 1)
    u = model(t)

u_pred = u.numpy()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_test, u_exact(t_test, d, w0), 'k:', label="Exact")
ax.plot(t_test, u_pred, 'r', lw=1, label="Prediction")
ax.set(xlabel="t", ylabel="u(t)")
ax.grid(color="k", ls=":", lw=1)
ax.legend()
plt.show()
```
