import os
import time

import torch

try:
    import wandb
except ImportError:
    wandb = None


def log_metrics(metrics):
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics)


def amp_context(device, amp_dtype):
    enabled = amp_dtype is not None and torch.device(device).type == "cuda"
    return torch.autocast(
        device_type=torch.device(device).type,
        dtype=amp_dtype,
        enabled=enabled,
    )


def train_epoch(
    model,
    criterion,
    device,
    dataloader,
    optimizer,
    scheduler=None,
    amp_dtype=None,
    gradient_clip=None,
):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for data, *target in dataloader:
        data = data.to(device, non_blocking=True)
        target = (
            [t.to(device, non_blocking=True) for t in target]
            if len(target) > 1
            else target[0].to(device, non_blocking=True)
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, amp_dtype):
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / num_batches


def val_epoch(model, criterion, device, dataloader, amp_dtype=None):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    with torch.inference_mode():
        for data, *target in dataloader:
            data = data.to(device, non_blocking=True)
            target = (
                [t.to(device, non_blocking=True) for t in target]
                if len(target) > 1
                else target[0].to(device, non_blocking=True)
            )
            with amp_context(device, amp_dtype):
                output = model(data)
                loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / num_batches


def train(
    epochs,
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    save_path,
    device,
    logger,
    val_iterations=1,
    val_iou_fn=None,
    val_iou_interval=1,
    amp_dtype=None,
    gradient_clip=None,
):
    """Trains the model and saves the best weights.

    Args:
        epochs (int): Number of epochs to train.
        dataloaders (tuple): Tuple containing the training and validation dataloaders.
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function to use.
        optimizer (torch.nn.Module): Optimizer to use.
        scheduler (torch.nn.Module): Learning rate scheduler to use.
        save_path (str): Path to save the best model weights.
        device (torch.device): Device to use.
        logger (logging.Logger): Logger to use.
        val_iterations (int, optional): Number of validation epochs to average. Defaults to 1.
    """
    train_loader, val_loader = dataloaders
    best_val_iou = -1.0
    best_val_loss = float("inf")
    history_path = os.path.join(save_path, "history.csv")
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_iou\n")
    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            criterion,
            device,
            train_loader,
            optimizer,
            scheduler=scheduler,
            amp_dtype=amp_dtype,
            gradient_clip=gradient_clip,
        )
        val_loss = 0
        if val_loader is not None:
            # each validation epoch is unique due to data augmentation, so we can average multiple
            for _ in range(val_iterations):
                val_loss += val_epoch(
                    model, criterion, device, val_loader, amp_dtype=amp_dtype
                )
            val_loss /= val_iterations
        val_iou = None
        if val_iou_fn is not None and (epoch + 1) % val_iou_interval == 0:
            val_iou = val_iou_fn(model)
        should_save = False
        if val_iou is not None and val_iou > best_val_iou:
            best_val_iou = val_iou
            should_save = True
        elif val_iou is None and epoch >= epochs * 0.9 and val_loss < best_val_loss:
            should_save = True
        if should_save:
            best_val_loss = val_loss
            state_model = getattr(model, "_orig_mod", model)
            torch.save(state_model.state_dict(), os.path.join(save_path, "best.pt"))
        message = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            + f" | EPOCH {(epoch+1):0{len(str(epochs))}}/{epochs}"
            + f" | TRAIN LOSS: {train_loss:.5f}"
            + f" | VAL LOSS: {val_loss:.5f}"
        )
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        if val_iou is not None:
            message += f" | VAL IOU: {val_iou:.5f}"
            metrics["val_iou"] = val_iou
        logger.info(message)
        log_metrics(metrics)
        with open(history_path, "a") as f:
            f.write(
                f"{epoch + 1},{train_loss:.8f},{val_loss:.8f},"
                + (f"{val_iou:.8f}" if val_iou is not None else "")
                + "\n"
            )
    log_metrics({"best_val_loss": best_val_loss, "best_val_iou": best_val_iou})
    return {"best_val_loss": best_val_loss, "best_val_iou": best_val_iou}
