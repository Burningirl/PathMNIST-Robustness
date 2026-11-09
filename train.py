"""PathMNIST script entry point. Implement SmallCNN in src/classification.py first."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PathMNIST
from src.classification import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.ToTensor()
    train = PathMNIST(split="train", download=True, transform=tfm, size=28)
    val = PathMNIST(split="val", download=True, transform=tfm, size=28)
    test = PathMNIST(split="test", download=True, transform=tfm, size=28)
    print(f"device={device}; train={len(train)} val={len(val)} test={len(test)}")
    print("TODO: implement model/training loop; select by validation macro-F1.")
    print("Evaluate clean and corrupted test data only after configuration is frozen.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
