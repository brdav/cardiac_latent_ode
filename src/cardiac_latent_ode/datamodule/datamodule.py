from typing import Callable, Literal

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class DataModule:
    def __init__(
        self,
        dataset_cls: Callable[..., Dataset],
        batch_size: int,
        num_workers: int = 6,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")

        self.dataset_train: Dataset | None = None
        self.dataset_val: Dataset | None = None
        self.dataset_test: Dataset | None = None

        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Literal["fit", "test"] = "fit") -> None:
        if stage == "fit":
            self.dataset_train = self.dataset_cls(split="train")
            self.dataset_val = self.dataset_cls(split="val")
            return

        if stage == "test":
            self.dataset_test = self.dataset_cls(split="test")
            return

        raise ValueError(f"Unsupported stage: {stage}")

    def _build_dataloader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError(
                "Requested dataloader before dataset initialization. "
                "Call setup('fit') for train/val or setup('test') for test first."
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.dataset_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.dataset_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(self.dataset_test, shuffle=False)
