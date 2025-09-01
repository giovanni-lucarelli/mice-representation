from torch.utils.data import Dataset


class AsTupleDataset(Dataset):
    """Wrap a dataset that returns dicts into a tuple (image, label)."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        if isinstance(item, dict):
            image = item.get('imgs', item.get('image'))
            label = item.get('lbls', item.get('label'))
            return image, label
        return item
