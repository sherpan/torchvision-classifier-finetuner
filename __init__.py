import os
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos
import fiftyone.utils.torch as fout

logger = logging.getLogger("fiftyone.core.collections")

TRAIN_ROOT = "/tmp/torchvision_classifier/"

SUPPORTED_MODELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b2": "EfficientNet-B2",
    "mobilenet_v3_large": "MobileNetV3-Large",
}


class FiftyOneClassificationDataset(Dataset):
    """Wraps fout.TorchImageDataset and pairs each image with its integer class label."""

    def __init__(self, view, label_field, class_to_idx, transform=None):
        # Build sample_id -> class_index map from the view
        self._label_map = {}
        for sample in view.iter_samples():
            label_obj = sample.get_field(label_field)
            if label_obj is None or label_obj.label is None:
                continue
            label_str = label_obj.label
            if label_str in class_to_idx:
                self._label_map[sample.id] = class_to_idx[label_str]

        # Filter view to only samples that have a valid label
        valid_ids = list(self._label_map.keys())
        valid_view = view.select(valid_ids)

        self._img_ds = fout.TorchImageDataset(
            samples=valid_view,
            include_ids=True,
            transform=transform,
            force_rgb=True,
            download=True,
        )

    def __len__(self):
        return len(self._img_ds)

    def __getitem__(self, idx):
        img, sample_id = self._img_ds[idx]
        return img, self._label_map[sample_id]


def build_model(model_name, num_classes, pretrained=True):
    """Load a torchvision model and replace its classification head."""
    import torchvision.models as models

    weights_arg = "DEFAULT" if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights_arg)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights_arg)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_transforms(img_size=224):
    """Standard ImageNet-style transforms for train and val."""
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


class TorchvisionClassifierFinetuner(foo.Operator):

    @property
    def config(self):
        return foo.OperatorConfig(
            name="torchvision-classifier-finetuner",
            label="Fine-tune Torchvision Classifier",
            description="Fine-tune a torchvision image classification model on a FiftyOne classification field",
            icon="model_training",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Fine-tune Torchvision Classifier",
                icon="model_training",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # --- Classification field selection ---
        dataset = ctx.dataset
        schema = dataset.get_field_schema(
            ftype=fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification,
        )
        field_names = list(schema.keys())

        field_choices = types.DropdownView()
        for name in field_names:
            field_choices.add_choice(name, label=name)

        inputs.enum(
            "label_field",
            field_choices.values(),
            required=True,
            label="Classification field",
            description="The FiftyOne field containing ground-truth classification labels",
            view=field_choices,
        )

        # --- Model selection ---
        model_choices = types.DropdownView()
        for model_key, model_label in SUPPORTED_MODELS.items():
            model_choices.add_choice(model_key, label=model_label)

        inputs.enum(
            "model_name",
            model_choices.values(),
            required=True,
            default="resnet50",
            label="Torchvision model",
            description="Pretrained model architecture to fine-tune",
            view=model_choices,
        )

        # --- Output path ---
        inputs.str(
            "export_uri",
            required=True,
            default="/tmp/torchvision_classifier/finetuned_model.pt",
            label="Output model path",
            description="Local or cloud (GCS/S3) path to save the fine-tuned model weights (.pt)",
        )

        # --- Hyperparameters ---
        inputs.int(
            "epochs",
            default=10,
            required=True,
            label="Epochs",
            description="Number of training epochs",
        )

        inputs.int(
            "batch_size",
            default=32,
            required=True,
            label="Batch size",
            description="Number of samples per training batch",
        )

        inputs.float(
            "learning_rate",
            default=1e-4,
            required=True,
            label="Learning rate",
            description="Initial learning rate for the optimizer",
        )

        inputs.float(
            "weight_decay",
            default=1e-4,
            required=False,
            label="Weight decay",
            description="L2 regularization strength (AdamW optimizer)",
        )

        inputs.int(
            "img_size",
            default=224,
            required=False,
            label="Image size",
            description="Input image size (height and width in pixels)",
        )

        inputs.int(
            "num_workers",
            default=4,
            required=False,
            label="DataLoader workers",
            description="Number of parallel workers for data loading",
        )

        inputs.int(
            "target_device_index",
            default=0,
            required=False,
            label="CUDA device index",
            description="CUDA GPU device number to use (ignored on MPS/CPU)",
        )

        return types.Property(
            inputs,
            view=types.View(label="Fine-tune Torchvision Classifier"),
        )

    def execute(self, ctx):
        label_field = ctx.params["label_field"]
        model_name = ctx.params["model_name"]
        export_uri = ctx.params["export_uri"]
        epochs = ctx.params["epochs"]
        batch_size = ctx.params["batch_size"]
        learning_rate = ctx.params["learning_rate"]
        weight_decay = ctx.params.get("weight_decay", 1e-4)
        img_size = ctx.params.get("img_size", 224)
        num_workers = ctx.params.get("num_workers", 4)
        target_device_index = ctx.params.get("target_device_index", 0)

        dataset = ctx.dataset

        # --- Resolve device ---
        cuda_count = torch.cuda.device_count()
        if cuda_count > 0:
            dev_idx = target_device_index if target_device_index < cuda_count else 0
            device = torch.device(f"cuda:{dev_idx}")
            logger.warning(f"Using CUDA device: cuda:{dev_idx}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.warning("Using Apple MPS (Metal) GPU")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU found, using CPU")

        # --- Discover classes ---
        label_key = f"{label_field}.label"
        classes = sorted(dataset.distinct(label_key))
        if not classes:
            raise ValueError(
                f"No labels found in field '{label_field}'. "
                "Ensure samples have Classification annotations."
            )
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)
        logger.warning(f"Found {num_classes} classes: {classes}")

        # --- Ensure train/val split ---
        all_tags = dataset.distinct("tags")
        if "train" not in all_tags or "val" not in all_tags:
            logger.warning("Missing train/val tags — auto-splitting 80/20...")
            ctx.set_progress(progress=0.02, label="Auto-splitting dataset 80/20...")
            sample_ids = dataset.values("id")
            random.seed(42)
            random.shuffle(sample_ids)
            split_idx = int(0.8 * len(sample_ids))
            dataset.select(sample_ids[:split_idx]).tag_samples("train")
            dataset.select(sample_ids[split_idx:]).tag_samples("val")
            logger.warning(
                f"Tagged {split_idx} train / {len(sample_ids) - split_idx} val samples"
            )

        # --- Build datasets & loaders ---
        ctx.set_progress(progress=0.05, label="Building data loaders...")
        train_transform, val_transform = get_transforms(img_size)

        train_ds = FiftyOneClassificationDataset(
            dataset.match_tags("train"), label_field, class_to_idx, train_transform
        )
        val_ds = FiftyOneClassificationDataset(
            dataset.match_tags("val"), label_field, class_to_idx, val_transform
        )

        logger.warning(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

        # --- Build model ---
        ctx.set_progress(progress=0.08, label="Loading pretrained model...")
        model = build_model(model_name, num_classes, pretrained=True)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # --- Log training configuration ---
        logger.warning(
            f"Starting training: model={model_name}, epochs={epochs}, "
            f"batch_size={batch_size}, lr={learning_rate}, weight_decay={weight_decay}, "
            f"img_size={img_size}, device={device}"
        )

        # --- Training loop ---
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

        # --- Save best checkpoint ---
        ctx.set_progress(progress=0.96, label="Saving model checkpoint...")

        os.makedirs(TRAIN_ROOT, exist_ok=True)
        local_ckpt = os.path.join(TRAIN_ROOT, "best_model.pt")

        torch.save(
            {
                "model_name": model_name,
                "num_classes": num_classes,
                "classes": classes,
                "class_to_idx": class_to_idx,
                "img_size": img_size,
                "state_dict": best_state,
                "best_val_acc": best_val_acc,
            },
            local_ckpt,
        )

        # Copy to final destination (supports local / GCS / S3 via fos)
        fos.copy_file(local_ckpt, export_uri)
        logger.warning(f"Saved best model (val_acc={best_val_acc:.4f}) to {export_uri}")

        ctx.set_progress(progress=1.0, label="Done!")
        return {
            "export_uri": export_uri,
            "best_val_acc": round(best_val_acc, 4),
            "num_classes": num_classes,
            "classes": ", ".join(classes),
            "status": "success",
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("export_uri", label="Model saved to")
        outputs.str("best_val_acc", label="Best validation accuracy")
        outputs.str("num_classes", label="Number of classes")
        outputs.str("classes", label="Classes")
        outputs.str("status", label="Status")
        return types.Property(
            outputs,
            view=types.View(label="Fine-tuning Results"),
        )


def register(plugin):
    plugin.register(TorchvisionClassifierFinetuner)
