import os
import sys
import random
import logging

# Ensure spawned DataLoader worker processes can import dataset.py.
# The plugin module name (@smehta73/torchvision-classifier-finetuner) contains
# characters that are not valid Python identifiers, so FiftyOneClassificationDataset
# lives in dataset.py instead. Adding the plugin directory to sys.path/PYTHONPATH
# lets workers do `import dataset` with a valid module name.
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)
_pypath = os.environ.get("PYTHONPATH", "")
if _plugin_dir not in _pypath.split(os.pathsep):
    os.environ["PYTHONPATH"] = _plugin_dir + (os.pathsep + _pypath if _pypath else "")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos
from dataset import FiftyOneClassificationDataset
from models import build_model, SUPPORTED_MODELS
from transforms import get_transforms
from trainer import train

logger = logging.getLogger("fiftyone.core.collections")

TRAIN_ROOT = "/tmp/torchvision_classifier/"


class TorchvisionClassifierFinetuner(foo.Operator):

    @property
    def config(self):
        return foo.OperatorConfig(
            name="torchvision-classifier-finetuner",
            label="Fine-tune Torchvision Classifier",
            description="Fine-tune a torchvision image classification model on a FiftyOne classification field",
            icon="model_training",
            dynamic=True,
            allow_immediate_execution=False,
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
            default="gs://my-bucket/torchvision_classifier/finetuned_model.pt",
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
            default=0,
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
        num_workers = ctx.params.get("num_workers", 0)
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

        # --- Pre-download cloud media so DataLoader workers hit local files ---
        ctx.set_progress(progress=0.05, label="Downloading media...")
        dataset.download_media()

        # --- Build datasets & loaders ---
        ctx.set_progress(progress=0.07, label="Building data loaders...")
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
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            multiprocessing_context="spawn" if num_workers > 0 else None,
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

        # --- Run training loop ---
        result = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       epochs, device, ctx)
        best_val_acc = result["best_val_acc"]
        best_state = result["best_state"]

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


class TorchvisionClassifierInference(foo.Operator):

    @property
    def config(self):
        return foo.OperatorConfig(
            name="torchvision-classifier-inference",
            label="Run Torchvision Classifier Inference",
            description="Run inference with a fine-tuned torchvision classification model on the current view",
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
                label="Run Torchvision Classifier Inference",
                icon="model_training",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        file_explorer = types.FileExplorerView(
            choose_dir=False,
            button_label="Choose model checkpoint",
            default_path="gs://my-bucket/torchvision_classifier/",
        )
        inputs.file(
            "model_uri",
            required=True,
            label="Model checkpoint",
            description="Local or cloud (GCS/S3) .pt checkpoint saved by the fine-tuner",
            view=file_explorer,
        )

        inputs.str(
            "label_field",
            required=True,
            default="predicted_label",
            label="Output label field",
            description="FiftyOne field name to write predicted Classification labels into",
        )

        inputs.int(
            "batch_size",
            default=64,
            required=False,
            label="Batch size",
            description="Number of images per inference batch (larger = faster on GPU)",
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
            view=types.View(label="Run Torchvision Classifier Inference"),
        )

    def execute(self, ctx):
        model_uri = ctx.params["model_uri"]["absolute_path"]
        label_field = ctx.params["label_field"]
        batch_size = ctx.params.get("batch_size", 64)
        num_workers = ctx.params.get("num_workers", 4)
        target_device_index = ctx.params.get("target_device_index", 0)

        view = ctx.view

        # --- Resolve device ---
        cuda_count = torch.cuda.device_count()
        if cuda_count > 0:
            dev_idx = target_device_index if target_device_index < cuda_count else 0
            device = torch.device(f"cuda:{dev_idx}")
            logger.warning(f"[inference] Using CUDA device: cuda:{dev_idx}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.warning("[inference] Using Apple MPS (Metal) GPU")
        else:
            device = torch.device("cpu")
            logger.warning("[inference] No GPU found, using CPU")

        # --- Load checkpoint ---
        ctx.set_progress(progress=0.02, label="Loading model checkpoint...")
        os.makedirs(TRAIN_ROOT, exist_ok=True)
        local_ckpt = os.path.join(TRAIN_ROOT, "inference_model.pt")
        fos.copy_file(model_uri, local_ckpt)

        ckpt = torch.load(local_ckpt, map_location="cpu", weights_only=True)
        model_name = ckpt["model_name"]
        classes = ckpt["classes"]
        img_size = ckpt["img_size"]
        num_classes = len(classes)

        logger.warning(
            f"[inference] Loaded checkpoint: model={model_name}, "
            f"classes={num_classes}, img_size={img_size}"
        )

        # --- Rebuild model and load weights ---
        ctx.set_progress(progress=0.05, label="Building model...")
        model = build_model(model_name, num_classes, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()

        # --- Build inference transform (val-style, no augmentation) ---
        _, infer_transform = get_transforms(img_size)

        # --- Pre-download cloud media so DataLoader workers hit local files ---
        ctx.set_progress(progress=0.08, label="Downloading media...")
        view.download_media()

        # --- Build DataLoader over the current view ---
        ctx.set_progress(progress=0.10, label="Building data loader...")
        import fiftyone.utils.torch as fout

        img_ds = fout.TorchImageDataset(
            samples=view,
            include_ids=True,
            transform=infer_transform,
            force_rgb=True,
            download=True,
        )

        loader = DataLoader(
            img_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            multiprocessing_context="spawn" if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
        )

        # --- Run inference ---
        ctx.set_progress(progress=0.12, label="Running inference...")
        total = len(img_ds)
        processed = 0
        pred_map = {}  # sample_id -> fo.Classification

        use_amp = device.type == "cuda"
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda")
            if use_amp
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )

        with torch.inference_mode(), autocast_ctx:
            for imgs, sample_ids in loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)

                for sid, pred_idx, conf in zip(sample_ids, preds.tolist(), confs.tolist()):
                    pred_map[sid] = fo.Classification(
                        label=classes[pred_idx],
                        confidence=conf,
                    )

                processed += len(sample_ids)
                ctx.set_progress(
                    progress=0.12 + (processed / max(total, 1)) * 0.83,
                    label=f"Inference: {processed}/{total} samples",
                )

        # --- Bulk-write predictions back to the dataset ---
        ctx.set_progress(progress=0.96, label="Saving predictions...")
        view_ids = view.values("id")
        classifications = [pred_map.get(sid) for sid in view_ids]
        view.set_values(label_field, classifications)

        ctx.set_progress(progress=1.0, label="Done!")
        ctx.trigger("reload_dataset")

        return {
            "label_field": label_field,
            "num_samples": processed,
            "model_name": model_name,
            "classes": ", ".join(classes),
            "status": "success",
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("label_field", label="Predictions written to")
        outputs.str("num_samples", label="Samples processed")
        outputs.str("model_name", label="Model")
        outputs.str("classes", label="Classes")
        outputs.str("status", label="Status")
        return types.Property(
            outputs,
            view=types.View(label="Inference Results"),
        )


def register(plugin):
    plugin.register(TorchvisionClassifierFinetuner)
    plugin.register(TorchvisionClassifierInference)
