import time
from copy import deepcopy


class EarlyStopper:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class EpochStopper:
    """Stop at specific epoch"""
    def __init__(self, max_epoch=10):
        self.max_epoch = max_epoch
        self.current_epoch = 0

    def __call__(self, val_loss, model):
        self.current_epoch += 1
        return self.current_epoch >= self.max_epoch


class LossThresholdStopper:
    """Stop when loss reaches target threshold"""
    def __init__(self, target_loss=0.001):
        self.target_loss = target_loss

    def __call__(self, val_loss, model):
        return val_loss <= self.target_loss


class MetricTargetStopper:
    """Stop when specific metric reaches target value"""
    def __init__(self, target_metrics={}, check_all=True):
        self.target_metrics = target_metrics  # {'ssim': 0.95, 'psnr': 30.0}
        self.check_all = check_all  # True: all metrics achieved, False: any metric achieved
        self.current_metrics = {}

    def update_metrics(self, metrics_dict):
        self.current_metrics = metrics_dict

    def __call__(self, val_loss, model):
        if not self.target_metrics or not self.current_metrics:
            return False

        achieved = []
        for metric_name, target_value in self.target_metrics.items():
            if metric_name in self.current_metrics:
                current_value = self.current_metrics[metric_name]
                # SSIM, PSNR etc. are better when higher
                if metric_name.lower() in ['ssim', 'psnr']:
                    achieved.append(current_value >= target_value)
                else:  # loss metrics are better when lower
                    achieved.append(current_value <= target_value)

        if self.check_all:
            return all(achieved)
        else:
            return any(achieved)


class TimeStopper:
    """Stop after specific time duration"""
    def __init__(self, max_time_hours=2.0):
        self.max_time_seconds = max_time_hours * 3600
        self.start_time = time.time()

    def __call__(self, val_loss, model):
        elapsed = time.time() - self.start_time
        return elapsed >= self.max_time_seconds


class PlateauStopper:
    """Stop when metric plateaus (more flexible than early stopping)"""
    def __init__(self, patience=10, min_delta=1e-4, mode='min',
                 restore_best_weights=True, factor=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'min' for loss, 'max' for accuracy/ssim
        self.restore_best_weights = restore_best_weights
        self.factor = factor
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_weights = None
        self.plateau_count = 0

    def __call__(self, val_loss, model):
        current = val_loss

        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
            self.plateau_count = 0
            if self.restore_best_weights:
                self.best_weights = deepcopy(model.state_dict())
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.plateau_count += 1
            self.counter = 0

            # Stop after 2 plateaus
            if self.plateau_count >= 2:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True

        return False


class CombinedStopper:
    """Combine multiple stopping criteria"""
    def __init__(self, stoppers=[], logic='or'):
        self.stoppers = stoppers
        self.logic = logic  # 'or': stop if any is True, 'and': stop if all are True

    def __call__(self, val_loss, model):
        results = [stopper(val_loss, model) for stopper in self.stoppers]

        if self.logic == 'or':
            return any(results)
        else:
            return all(results)


def get_stopper(stopper_type, **stopper_params):
    """Factory function to create stopping strategies"""
    available_stoppers = [
        'early_stop', 'stop', 'loss_threshold', 'metric_target',
        'time', 'plateau', 'combined', 'none'
    ]
    stopper_type = stopper_type.lower()

    if stopper_type == 'early_stop':
        params = {'patience': 10, 'min_delta': 0.0, 'restore_best_weights': True}
        params.update(stopper_params)
        return EarlyStopper(**params)

    elif stopper_type == 'stop':
        params = {'max_epoch': 10}
        params.update(stopper_params)
        return EpochStopper(**params)

    elif stopper_type == 'loss_threshold':
        params = {'target_loss': 0.001}
        params.update(stopper_params)
        return LossThresholdStopper(**params)

    elif stopper_type == 'metric_target':
        params = {'target_metrics': {'ssim': 0.95}, 'check_all': True}
        params.update(stopper_params)
        return MetricTargetStopper(**params)

    elif stopper_type == 'time':
        params = {'max_time_hours': 2.0}
        params.update(stopper_params)
        return TimeStopper(**params)

    elif stopper_type == 'plateau':
        params = {
            'patience': 10, 'min_delta': 1e-4, 'mode': 'min',
            'restore_best_weights': True, 'factor': 0.5
        }
        params.update(stopper_params)
        return PlateauStopper(**params)

    elif stopper_type == 'combined':
        params = {'stoppers': [], 'logic': 'or'}
        params.update(stopper_params)
        return CombinedStopper(**params)

    elif stopper_type == 'none':
        return None

    else:
        raise ValueError(f"Unknown stopper type: {stopper_type}. Available stoppers: {available_stoppers}")


if __name__ == "__main__":
    # Test code
    print("Testing stopper creation...")

    # Early stopper
    early_stopper = get_stopper('early_stop', patience=5, min_delta=1e-4)
    print(f"Early stopper created: patience={early_stopper.patience}")

    # Fixed epoch stopper
    epoch_stopper = get_stopper('stop', max_epoch=10)
    print(f"Epoch stopper created: max_epoch={epoch_stopper.max_epoch}")

    # Loss threshold stopper
    loss_stopper = get_stopper('loss_threshold', target_loss=0.01)
    print(f"Loss threshold stopper created: target={loss_stopper.target_loss}")

    # Metric target stopper
    metric_stopper = get_stopper('metric_target',
        target_metrics={'ssim': 0.95, 'psnr': 30.0},
        check_all=True
    )
    print(f"Metric target stopper created: targets={metric_stopper.target_metrics}")

    # Time-based stopper
    time_stopper = get_stopper('time', max_time_hours=2.0)
    print(f"Time stopper created: max_time={time_stopper.max_time_seconds}s")

    # Combined stopper
    combined_stopper = get_stopper('combined',
        stoppers=[early_stopper, time_stopper],
        logic='or'
    )
    print(f"Combined stopper created with {len(combined_stopper.stoppers)} stoppers")

    print("All stoppers created successfully!")