# torchvision-classifier-finetuner

A [FiftyOne plugin](https://docs.voxel51.com/plugins/index.html) operator that fine-tunes a pretrained torchvision image classification model on any labeled FiftyOne dataset — directly from the UI or Python API, no training boilerplate required.

## Features

- Fine-tune ResNet-50, EfficientNet-B2, or MobileNetV3-Large on any FiftyOne `Classification` field
- Auto train/val split, configurable hyperparameters, and best-checkpoint saving
- Export to local, GCS, or S3 paths
- Uses `FiftyOneClassificationDataset` (`dataset.py`) — a lightweight `torch.utils.data.Dataset` that wraps [`fout.TorchImageDataset`](https://docs.voxel51.com/api/fiftyone.utils.torch.html), automatically filters samples with missing labels, and returns `(image_tensor, class_index)` pairs ready for training
- Modular file layout: model building (`models.py`), data augmentation (`transforms.py`), and the training loop (`trainer.py`) are each in their own focused module — making the plugin easy to extend without touching unrelated code

## What it does

The plugin performs transfer learning on top of a pretrained image classification backbone. Given a FiftyOne dataset with a `Classification` label field, it will:

1. Discover all unique classes in the label field.
2. Auto-create an 80/20 train/val split (by tagging samples) if one doesn't already exist.
3. Load a pretrained torchvision model and replace the final classification head with one sized for your classes.
4. Train with AdamW + CosineAnnealingLR for the specified number of epochs, saving the best checkpoint by validation accuracy.
5. Export the checkpoint to a local path, GCS (`gs://…`), or S3 (`s3://…`).

The returned checkpoint is a `.pt` file containing the weights plus metadata (architecture name, class labels, image size) so it can be reloaded for inference without re-specifying those details.

---

## Dataset requirements
- The specified `label_field` must contain `fo.Classification` labels.
- If your dataset already has `"train"` and `"val"` tags on samples, those splits will be used. Otherwise the plugin automatically tags 80% as `"train"` and 20% as `"val"`.

---

## Usage

### From the FiftyOne UI

1. Open a dataset with a `Classification` label field.
2. Click the **Fine-tune Classifier** button in the Samples Grid secondary actions bar.
3. Fill in the input form and click **Schedule** to run the fine-tuning job as a delegated operator.

### From Python

```python
import fiftyone as fo
import fiftyone.operators as foo

dataset = fo.load_dataset("my_dataset")

op = foo.get_operator("@smehta73/torchvision-classifier-finetuner")
op.execute(
    fo.OperatorExecutionContext(
        dataset=dataset,
        params={
            "label_field": "ground_truth",
            "model_name": "resnet50",
            "export_uri": "/tmp/my_model.pt",
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
    )
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_field` | string | — | The `Classification` field on your dataset to train on |
| `model_name` | choice | `resnet50` | Backbone architecture (see supported models below) |
| `export_uri` | string | — | Output path for the `.pt` checkpoint (local, `gs://`, or `s3://`) |
| `epochs` | int | 10 | Number of training epochs |
| `batch_size` | int | 32 | Mini-batch size |
| `learning_rate` | float | 1e-4 | Initial learning rate for AdamW |
| `weight_decay` | float | 1e-4 | L2 regularization coefficient |
| `img_size` | int | 224 | Input image size (square, in pixels) |
| `num_workers` | int | 0 | DataLoader worker processes |
| `target_device_index` | int | 0 | CUDA GPU index (ignored if no GPU is present) |

### Supported model architectures

| `model_name` value | Architecture |
|--------------------|--------------|
| `resnet50` | ResNet-50 |
| `efficientnet_b2` | EfficientNet-B2 |
| `mobilenet_v3_large` | MobileNetV3-Large |

---



## Customizing for your use case

The plugin is split into focused modules — each covering one concern. Edit only the file relevant to what you want to change:

| Goal | File to edit |
|---|---|
| Add a new backbone (ViT, ConvNeXt, etc.) | `models.py` |
| Change data augmentation | `transforms.py` |
| Swap loss function (label smoothing, focal loss) | `__init__.py` — one line in `execute()` |
| Swap optimizer or LR scheduler | `__init__.py` — one line in `execute()` |
| Change train/val split ratio or strategy | `__init__.py` — `execute()` split section |
| Change how samples are filtered or loaded | `dataset.py` |
| Run the training loop standalone (no FiftyOne) | Import and call `trainer.train()` directly |

### File responsibilities

- **`models.py`** — `build_model()` + `SUPPORTED_MODELS` dict. The UI dropdown auto-populates from `SUPPORTED_MODELS`, so adding a key here is all it takes to expose a new architecture.
- **`transforms.py`** — `get_transforms()`. Augmentation changes stay fully isolated from training logic.
- **`trainer.py`** — `train()` function. Accepts model, loaders, criterion, optimizer, scheduler, epochs, device, and ctx. Returns `best_val_acc` and `best_state`. Can be imported and called outside of FiftyOne.
- **`dataset.py`** — `FiftyOneClassificationDataset`. Handles label filtering and mapping between FiftyOne sample IDs and integer class indices.
- **`__init__.py`** — Thin operator shell. `execute()` wires together the modules: discovers classes, handles the train/val split, builds dataloaders, constructs criterion/optimizer/scheduler, calls `trainer.train()`, and saves the checkpoint.

### Add a new model backbone

Edit `models.py`:

```python
# models.py
SUPPORTED_MODELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b2": "EfficientNet-B2",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "vit_b_16": "ViT-B/16",   # <-- add entry here
}

def build_model(model_name, num_classes, pretrained=True):
    ...
    if model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    # ... existing branches below
```

### Change data augmentation

Edit `transforms.py`:

```python
def get_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),               # swap in RandAugment
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
```

### Change the loss function

In `execute()` in `__init__.py`:

```python
# e.g. label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Change the optimizer or learning rate schedule

In `execute()` in `__init__.py`:

```python
# Example: switch to SGD with StepLR
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

### Freeze the backbone (linear-probe style)

In `execute()` in `__init__.py`, after `build_model()`:

```python
model = build_model(model_name, num_classes)
for name, param in model.named_parameters():
    if "fc" not in name and "classifier" not in name and "heads" not in name:
        param.requires_grad = False
```

### Change the train/val split ratio

In `execute()` in `__init__.py`:

```python
train_ratio = 0.8   # <-- adjust this
```

### Customize how samples are loaded

Edit the constructor loop in `dataset.py`:

```python
for sample in view.iter_samples():
    label_obj = sample.get_field(label_field)
    if label_obj is None or label_obj.label is None:
        continue
    # Example: skip samples with confidence below a threshold
    if label_obj.confidence is not None and label_obj.confidence < 0.9:
        continue
    label_str = label_obj.label
    if label_str in class_to_idx:
        self._label_map[sample.id] = class_to_idx[label_str]
```

### Use the trainer standalone (without FiftyOne operator)

`trainer.train()` has no FiftyOne operator dependency beyond `ctx.set_progress`. Pass a lightweight context stub if running outside the plugin:

```python
from trainer import train

class DummyCtx:
    def set_progress(self, progress, label=""): print(f"{progress:.0%} {label}")

result = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
               epochs=10, device=device, ctx=DummyCtx())
best_state = result["best_state"]
best_val_acc = result["best_val_acc"]
```

---
## License

MIT
