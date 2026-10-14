from import_config import GlobalConfig
from pathlib import Path

from torchvision import datasets, transforms
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import json

import matplotlib.pyplot as plt

from utils import set_all_seed
from IJEPA import IJEPA
from metrics_tracker import MetricTracker



CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

DATA_ROOT = Path(__file__).resolve().parent / "data"
RUNS_ROOT = Path(__file__).resolve().parent / "runs"

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD
    )
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD
    )
])


def main(config):
    set_all_seed(13)
    device = torch.device(config.device)
    
    train_set = datasets.CIFAR10(
        root      = DATA_ROOT,
        train     = True,
        download  = True,
        transform = train_transform
    )

    train_loader = DataLoader(
        dataset    = train_set,
        batch_size = config.training.batch_size,
        shuffle    = True,
        drop_last  = True
    )

    model = IJEPA(
        device=config.device, 
        config=config.model
    )

    tracker = MetricTracker(RUNS_ROOT, config)

    step = 0
    for epoch in range(config.training.num_epochs):
        print(f"\n --- Epoch {epoch} --- ")
        for batch, _ in train_loader:
            metrics = model.train(batch.to(device))
            step   += 1
            if step % config.training.save_metrics_every == 0:
                tracker.log(step, metrics)
                print(f"--- Gradient steps : {step} ---")
                print(f"Loss : {metrics["loss"]}")
    tracker.close()
            


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "configs" / "IJEPA_CIFAR10_config.yaml"
    config = GlobalConfig.from_yaml(config_path)
    main(config=config)