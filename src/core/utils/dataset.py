from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
