import logging

import torch

logger = logging.getLogger("fiftyone.core.collections")


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, ctx) -> dict:
    """Run the training loop and return the best checkpoint state.

    Args:
        model: The PyTorch model to train (already on device).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        criterion: Loss function (e.g. nn.CrossEntropyLoss()).
        optimizer: Optimizer (e.g. AdamW).
        scheduler: LR scheduler (e.g. CosineAnnealingLR).
        epochs: Number of training epochs.
        device: torch.device to run training on.
        ctx: Operator context providing ctx.set_progress(progress, label).

    Returns:
        dict with keys:
            "best_val_acc" (float): Best validation accuracy achieved.
            "best_state" (dict): model.state_dict() snapshot at best val acc.
    """
    best_val_acc = 0.0
    best_state = None
    num_train_batches = len(train_loader)
    log_interval = max(1, num_train_batches // 4)  # log ~4 times per epoch

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (batch_imgs, batch_labels) in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()
            total += batch_imgs.size(0)

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_train_batches:
                batch_loss = running_loss / max(total, 1)
                batch_acc = correct / max(total, 1)
                logger.warning(
                    f"  Epoch {epoch + 1}/{epochs} "
                    f"[{batch_idx + 1}/{num_train_batches}] "
                    f"loss: {batch_loss:.4f}, acc: {batch_acc:.3f}"
                )

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(batch_labels).sum().item()
                val_total += batch_imgs.size(0)

        val_acc = val_correct / max(val_total, 1)
        val_loss = val_loss / max(val_total, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        new_best = val_acc > best_val_acc
        if new_best:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        progress = 0.1 + (epoch + 1) / epochs * 0.85
        label = (
            f"Epoch {epoch + 1}/{epochs} — "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.3f}, "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.3f}"
        )
        ctx.set_progress(progress=progress, label=label)
        best_marker = " [NEW BEST]" if new_best else ""
        logger.warning(f"{label}, lr: {current_lr:.2e}{best_marker}")

    return {"best_val_acc": best_val_acc, "best_state": best_state}
