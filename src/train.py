# train.py
# Purpose: Training loop for DrumClassifier with checkpointing
# Features: Train/test split, epoch logging, best-model save, loss/F1 plots
# Usage: python src/train.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import build_dataloaders
from src.model import DrumClassifier


# The notebook model is trained with MSE against normalized 0-1 velocity targets.
# Clamp predictions only for reporting metrics so raw model outputs do not distort F1 counting.
def _metric_predictions(outputs):
    return torch.clamp(outputs, 0.0, 1.0)


# Returns a sibling checkpoint path for alternate selection criteria from the same training run.
def _derive_checkpoint_path(model_save_path, suffix):
    root, ext = os.path.splitext(model_save_path)
    return f"{root}_{suffix}{ext}"


# Full training run: builds data pipeline, trains model for NUM_EPOCHS,
# saves best checkpoint by test loss, and generates loss/F1 plots.
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    print(f"Using device: {device}")

    train_loader, test_loader, dataset = build_dataloaders()
    print(f"Dataset size: {len(dataset)}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = DrumClassifier().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    start_epoch = 0
    train_losses = []
    train_f1s = []
    test_losses = []
    test_f1s = []
    best_test_loss = float("inf")
    best_test_f1 = float("-inf")
    model_save_path = None
    best_f1_model_save_path = None

    # Resume full training state if an interrupted run left a checkpoint.
    # This is separate from the timestamped best-model files, which are the
    # durable outputs we keep after training completes.
    if os.path.exists(config.CHECKPOINT_PATH):
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_test_loss = checkpoint["best_test_loss"]
        best_test_f1 = checkpoint.get("best_test_f1", float("-inf"))
        train_losses = checkpoint["train_losses"]
        test_losses = checkpoint["test_losses"]
        train_f1s = checkpoint["train_f1s"]
        test_f1s = checkpoint["test_f1s"]
        model_save_path = checkpoint["model_save_path"]
        best_f1_model_save_path = checkpoint.get(
            "best_f1_model_save_path",
            _derive_checkpoint_path(model_save_path, "best_f1"),
        )
        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Model will be saved to: {model_save_path}")
        print(f"Best-F1 model will be saved to: {best_f1_model_save_path}")
    else:
        model_save_path = config.get_model_save_path()
        best_f1_model_save_path = _derive_checkpoint_path(model_save_path, "best_f1")
        print("Starting fresh training run")
        print(f"Model will be saved to: {model_save_path}")
        print(f"Best-F1 model will be saved to: {best_f1_model_save_path}")

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        tp_train, fp_train, fn_train = 0.0, 0.0, 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_VALUE)
            optimizer.step()

            epoch_train_loss += loss.item()
            # Training optimizes dense heatmap regression, but reporting treats
            # each note/time cell as a hit/no-hit decision using the same
            # thresholding convention as inference.
            metric_outputs = _metric_predictions(outputs)
            pred_hits = (metric_outputs > config.HIT_THRESHOLD).float()
            true_hits = (targets > config.HIT_THRESHOLD).float()
            tp_train += (pred_hits * true_hits).sum().item()
            fp_train += (pred_hits * (1 - true_hits)).sum().item()
            fn_train += ((1 - pred_hits) * true_hits).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_precision = tp_train / (tp_train + fp_train + 1e-8)
        train_recall = tp_train / (tp_train + fn_train + 1e-8)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-8)
        train_losses.append(avg_train_loss)
        train_f1s.append(train_f1)

        model.eval()
        epoch_test_loss = 0.0
        tp_test, fp_test, fn_test = 0.0, 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                epoch_test_loss += loss.item()
                metric_outputs = _metric_predictions(outputs)
                pred_hits = (metric_outputs > config.HIT_THRESHOLD).float()
                true_hits = (targets > config.HIT_THRESHOLD).float()
                tp_test += (pred_hits * true_hits).sum().item()
                fp_test += (pred_hits * (1 - true_hits)).sum().item()
                fn_test += ((1 - pred_hits) * true_hits).sum().item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_precision = tp_test / (tp_test + fp_test + 1e-8)
        test_recall = tp_test / (tp_test + fn_test + 1e-8)
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall + 1e-8)

        scheduler.step(avg_test_loss)

        test_losses.append(avg_test_loss)
        test_f1s.append(test_f1)

        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}], "
            f"Loss: {avg_train_loss:.4e} / {avg_test_loss:.4e}, "
            f"F1: {train_f1:.4f} / {test_f1:.4f}, "
            f"P: {train_precision:.2f} / {test_precision:.2f}, "
            f"R: {train_recall:.3f} / {test_recall:.3f}"
        )

        # Preserve both selection criteria because the lowest-loss epoch and the
        # highest event-level F1 epoch are not guaranteed to be the same run state.
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Saved best model (test loss: {best_test_loss:.4e})")

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            torch.save(model.state_dict(), best_f1_model_save_path)
            print(f"  -> Saved best F1 model (test F1: {best_test_f1:.4f})")

        # Save resumable training state after every epoch so interrupted runs can
        # continue without losing optimizer, scheduler, or metric history.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_test_loss": best_test_loss,
                "best_test_f1": best_test_f1,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "train_f1s": train_f1s,
                "test_f1s": test_f1s,
                "model_save_path": model_save_path,
                "best_f1_model_save_path": best_f1_model_save_path,
            },
            config.CHECKPOINT_PATH,
        )

    if os.path.exists(config.CHECKPOINT_PATH):
        os.remove(config.CHECKPOINT_PATH)
        print("Checkpoint file removed (training complete)")

    os.makedirs(config.FILES_DIR, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig(os.path.join(config.FILES_DIR, "training_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(train_f1s, label="Training F1")
    plt.plot(test_f1s, label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("F1 over Epochs")
    plt.savefig(os.path.join(config.FILES_DIR, "training_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nTraining complete. Best test loss: {best_test_loss:.4e}")
    print(f"Best test F1: {best_test_f1:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"Best-F1 model saved to: {best_f1_model_save_path}")
    print(f"Plots saved to: {config.FILES_DIR}/")


if __name__ == "__main__":
    train()
