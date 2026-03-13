import fiftyone as fo
import fiftyone.operators as foo

fo.config.plugins_dir = "/Users/siddharthmehta/sid_foe/foe_plugins"

# ── Configuration ─────────────────────────────────────────────────────────────
OPERATOR_URI    = "@smehta73/torchvision-classifier-finetuner/torchvision-classifier-finetuner"
DATASET_NAME    = "pannuke"

LABEL_FIELD     = "tissue_type"        # field containing class labels
MODEL_NAME      = "resnet50"          # resnet50 | efficientnet_b2 | mobilenet_v3_large
EXPORT_URI      = "/tmp/torchvision_classifier/test_resnet50.pt"
EPOCHS          = 3
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
IMG_SIZE        = 224                  # cifar10 is 32x32
NUM_WORKERS     = 0
DEVICE_INDEX    = 0
# ──────────────────────────────────────────────────────────────────────────────

dataset = fo.load_dataset(DATASET_NAME)

ctx = {
    "view": dataset.view(),
    "params": {
        "label_field":        LABEL_FIELD,
        "model_name":         MODEL_NAME,
        "export_uri":         EXPORT_URI,
        "epochs":             EPOCHS,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LEARNING_RATE,
        "weight_decay":       WEIGHT_DECAY,
        "img_size":           IMG_SIZE,
        "num_workers":        NUM_WORKERS,
        "target_device_index": DEVICE_INDEX,
    },
}

result = foo.execute_operator(OPERATOR_URI, ctx)

print(result.result)
assert result.result["status"] == "success"
assert result.result["num_classes"] == 19
