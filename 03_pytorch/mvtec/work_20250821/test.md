### main.py

```python
model_names = ["autoencoder", "vae", "stfpm"]

for model_name in model_names:
    model = get_model(model_name)
    optimizer = get_optimizer(model_name)
    loss_fn = get_loss_fn(model_name)
    metrics = get_metrics(model_name)

    trainer = Trainer(model, optimizer, loss_fn, metrics=metrics)
    history = trainer.fit(train_loader, num_epochs, valid_loader=valid_loader)
```
