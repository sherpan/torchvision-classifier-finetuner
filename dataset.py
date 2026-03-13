from torch.utils.data import Dataset
import fiftyone.utils.torch as fout


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
