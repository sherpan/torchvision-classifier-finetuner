# torchvision-classifier-finetuner

A [FiftyOne plugin](https://docs.voxel51.com/plugins/index.html) operator that fine-tunes a pretrained torchvision image classification model on any labeled FiftyOne dataset — directly from the UI or Python API, no training boilerplate required.

## What it does

The plugin performs transfer learning on top of a pretrained image classification backbone. Given a FiftyOne dataset with a `Classification` label field, it will:

1. Discover all unique classes in the label field.
2. Auto-create an 80/20 train/val split (by tagging samples) if one doesn't already exist.
3. Load a pretrained torchvision model and replace the final classification head with one sized for your classes.
4. Train with AdamW + CosineAnnealingLR for the specified number of epochs, saving the best checkpoint by validation accuracy.
5. Export the checkpoint to a local path, GCS (`gs://…`), or S3 (`s3://…`).

The returned checkpoint is a `.pt` file containing the weights plus metadata (architecture name, class labels, image size) so it can be reloaded for inference without re-specifying those details.

---

## Installation

```bash
fiftyone plugins download https://github.com/smehta73/torchvision-classifier-finetuner
```

Or clone and install locally:

```bash
git clone https://github.com/smehta73/torchvision-classifier-finetuner
fiftyone plugins local-install torchvision-classifier-finetuner
```

### Dependencies

```bash
pip install torch torchvision fiftyone Pillow
```

---

## Usage

### From the FiftyOne UI

1. Open a dataset with a `Classification` label field.
2. Click the **Fine-tune Classifier** button in the Samples Grid secondary actions bar.
3. Fill in the input form and click **Execute** (or **Schedule** for delegated background execution).

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
| `num_workers` | int | 4 | DataLoader worker processes |
| `target_device_index` | int | 0 | CUDA GPU index (ignored if no GPU is present) |

### Supported model architectures

| `model_name` value | Architecture |
|--------------------|--------------|
| `resnet50` | ResNet-50 |
| `efficientnet_b2` | EfficientNet-B2 |
| `mobilenet_v3_large` | MobileNetV3-Large |

---

## Dataset requirements

- Samples must have image filepaths that exist on disk.
- The specified `label_field` must contain `fo.Classification` labels.
- If your dataset already has `"train"` and `"val"` tags on samples, those splits will be used. Otherwise the plugin automatically tags 80% as `"train"` and 20% as `"val"`.

---

## Customizing for your use case

All customization lives in `__init__.py`. Below are the most common changes and exactly where to make them.

### Add a new model backbone

Edit the `SUPPORTED_MODELS` dict near the top of `__init__.py` and the `build_model()` function:

```python
# __init__.py  ~line 18
SUPPORTED_MODELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b2": "EfficientNet-B2",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "vit_b_16": "ViT-B/16",          # <-- add your entry here
}

# __init__.py  build_model()  ~line 59
def build_model(model_name, num_classes):
    if model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model
    # ... existing branches below
```

### Change data augmentation

Edit `get_transforms()` (~line 82). The training pipeline currently uses `RandomResizedCrop`, `RandomHorizontalFlip`, and `ColorJitter`. Add or replace transforms here:

```python
def get_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),        # add vertical flip
            transforms.RandAugment(),               # swap in RandAugment
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    # ... validation branch unchanged
```

### Change the optimizer or learning rate schedule

In the `execute()` method (~line 323):

```python
# Current
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Example: switch to SGD with StepLR
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

### Freeze the backbone (linear-probe style)

After `build_model()` is called in `execute()`, freeze all layers except the classification head:

```python
model = build_model(model_name, num_classes)
for name, param in model.named_parameters():
    if "fc" not in name and "classifier" not in name and "heads" not in name:
        param.requires_grad = False
```

### Change the train/val split ratio

The auto-split logic is in `execute()` (~line 280). Change `0.8` to your desired training fraction:

```python
train_ratio = 0.8   # <-- adjust this
```

### Use a different loss function

In the training loop (~line 350), replace `nn.CrossEntropyLoss` with any PyTorch loss:

```python
# e.g. label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## Output checkpoint format

The saved `.pt` file is a plain Python dict loadable with `torch.load()`:

```python
checkpoint = torch.load("my_model.pt")
# Keys:
# checkpoint["model_name"]      -> e.g. "resnet50"
# checkpoint["num_classes"]     -> int
# checkpoint["classes"]         -> list of class label strings
# checkpoint["class_to_idx"]    -> dict mapping label -> int index
# checkpoint["img_size"]        -> int
# checkpoint["state_dict"]      -> OrderedDict of model weights
# checkpoint["best_val_acc"]    -> float (0–1)
```

### Loading the checkpoint for inference

```python
import torch, torchvision
import torch.nn as nn

ckpt = torch.load("my_model.pt")
model = getattr(torchvision.models, ckpt["model_name"])(weights=None)

# Replace head (same logic as build_model)
if hasattr(model, "fc"):
    model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])
elif hasattr(model, "classifier"):
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, ckpt["num_classes"])

model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## License

MIT
