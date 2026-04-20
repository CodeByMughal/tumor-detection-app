import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full pass through the training data."""
    model.train()

    total_loss    = 0.0
    correct       = 0
    total         = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────
        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # get predictions
        loss    = criterion(outputs, labels)  # calculate error

        # ── Backward pass ─────────────────────────────────────────
        loss.backward()                # compute gradients
        optimizer.step()               # update weights

        # ── Track stats ───────────────────────────────────────────
        total_loss += loss.item()
        preds       = torch.argmax(outputs, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Evaluate model on validation set — no learning happens here."""
    model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():              # no gradients needed
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            preds       = torch.argmax(outputs, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader,
                num_epochs=20, learning_rate=0.001,
                device='cpu', save_dir='ai_model/saved_models',
                model_name='tumor_model'):
    """
    Full training loop with:
    - Loss + accuracy tracking
    - Best model saving
    - Learning rate scheduling
    - Training history plot
    """

    os.makedirs(save_dir, exist_ok=True)

    # ── Loss function & optimizer ─────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=1e-4)

    # Reduce learning rate when accuracy stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    # ── Training history ──────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }

    best_val_acc  = 0.0
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')

    print(f"\n{'='*55}")
    print(f" Training: {model_name}")
    print(f" Epochs:   {num_epochs}  |  LR: {learning_rate}  |  Device: {device}")
    print(f"{'='*55}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
          f"{'Val Loss':>9} {'Val Acc':>9} {'':>8}")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):

        # ── Train ──────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # ── Validate ───────────────────────────────────────────────
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # ── Scheduler step ─────────────────────────────────────────
        scheduler.step(val_acc)

        # ── Save best model ────────────────────────────────────────
        flag = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc':    val_acc,
                'val_loss':   val_loss,
            }, best_model_path)
            flag = '  ← best'

        # ── Log progress ───────────────────────────────────────────
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.1f}% "
              f"{val_loss:>9.4f} {val_acc:>8.1f}%{flag}")

        # ── Store history ──────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Best model saved to: {best_model_path}")

    # ── Plot training curves ───────────────────────────────────────
    plot_training_history(history, model_name)

    return history, best_val_acc


def plot_training_history(history, model_name='model'):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train loss',    markersize=4)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val loss',      markersize=4)
    ax1.set_title('Loss over epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Val accuracy',   markersize=4)
    ax2.set_title('Accuracy over epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Training history — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'ai_model/saved_models/{model_name}_training.png')
    plt.show()
    print(f"Training plot saved!")