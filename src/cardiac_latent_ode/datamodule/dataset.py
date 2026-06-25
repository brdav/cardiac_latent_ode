import random
import warnings
from pathlib import Path
from typing import Any, Literal

import h5py
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from cardiac_latent_ode.utils.constants import TIME_FRAMES, EPS


class MeshDataset(Dataset):
    """Disk-backed mesh dataset backed by a single HDF5 file."""

    def __init__(
        self,
        split: Literal["train", "val", "test", "all"],
        h5_data_path: str,
        cohort_file_path: str,
        transform: Any = None,
        pre_transform: Any = None,
        stats_max_cases: int = 1000,
        val_fraction: float = 0.10,
    ) -> None:
        self.h5_data_path = h5_data_path
        self.cohort_file_path = cohort_file_path
        self.transform = transform
        self.stats_max_cases = stats_max_cases
        self.val_fraction = val_fraction
        root_dir = Path(h5_data_path).parent

        super().__init__(str(root_dir), transform, pre_transform)

        train_ids, test_ids = self._read_cohort_splits()
        case_ids = self._read_case_ids()
        train_files, val_files, test_files = self._build_split_indices(
            case_ids, train_ids, test_ids
        )
        self.file_list = self._select_files_for_dtype(
            split, train_files, val_files, test_files
        )

        stats_cache_path = root_dir / f"{self.stats_max_cases}.pt"

        stats_signature = {
            "train_count": int(len(train_files)),
            "val_count": int(len(val_files)),
            "test_count": int(len(test_files)),
            "stats_max_cases": int(self.stats_max_cases),
            "covariates": ["sex", "age", "bsa", "heart_rate"],
        }

        loaded_cached_stats = self._load_cached_stats(stats_cache_path, stats_signature)
        if not loaded_cached_stats:
            self._compute_and_cache_stats(
                train_files, stats_cache_path, stats_signature
            )

        self.data_handle = None

    def _read_cohort_splits(self) -> tuple[list[str], list[str]]:
        cohort_df = pd.read_csv(self.cohort_file_path, dtype=str)
        required_columns = {"case_id", "split"}
        missing_columns = required_columns - set(cohort_df.columns)
        if missing_columns:
            raise ValueError(
                f"Cohort file is missing required columns: {sorted(missing_columns)}"
            )

        train_ids = (
            cohort_df.loc[cohort_df["split"] == "train", "case_id"].astype(str).tolist()
        )
        test_ids = (
            cohort_df.loc[cohort_df["split"] == "test", "case_id"].astype(str).tolist()
        )
        return train_ids, test_ids

    def _read_case_ids(self) -> list[str]:
        with h5py.File(self.h5_data_path, "r") as h5f:
            case_id_ds = self._get_dataset(h5f, "case_id")
            return [self._decode_string(case_id) for case_id in case_id_ds[:]]

    def _build_split_indices(
        self,
        case_ids: list[str],
        train_ids: list[str],
        test_ids: list[str],
    ) -> tuple[list[int], list[int], list[int]]:
        id_to_indices: dict[str, list[int]] = {}
        for idx, case_id in sorted(enumerate(case_ids), key=lambda x: x[1]):
            id_to_indices.setdefault(case_id, []).append(idx)

        train_files, missing_train = self._resolve_indices(train_ids, id_to_indices)
        test_files, missing_test = self._resolve_indices(test_ids, id_to_indices)

        if missing_train:
            warnings.warn(
                f"{missing_train} train case_ids from cohort file are missing in the HDF5 dataset.",
                stacklevel=2,
            )
        if missing_test:
            warnings.warn(
                f"{missing_test} test case_ids from cohort file are missing in the HDF5 dataset.",
                stacklevel=2,
            )

        random.shuffle(train_files)
        n_val = int(round(self.val_fraction * len(train_files)))
        val_files = train_files[:n_val]
        train_files = train_files[n_val:]
        return train_files, val_files, test_files

    @staticmethod
    def _resolve_indices(
        case_ids: list[str], id_to_indices: dict[str, list[int]]
    ) -> tuple[list[int], int]:
        resolved = []
        missing = 0
        for case_id in case_ids:
            indices = id_to_indices.get(case_id)
            if indices is None:
                missing += 1
                continue
            resolved.extend(indices)
        return resolved, missing

    @staticmethod
    def _select_files_for_dtype(
        dtype: str,
        train_files: list[int],
        val_files: list[int],
        test_files: list[int],
    ) -> list[int]:
        if dtype == "train":
            return train_files
        if dtype == "val":
            return val_files
        if dtype == "test":
            return test_files
        if dtype == "all":
            return train_files + val_files + test_files
        raise ValueError(f"Invalid dtype: {dtype}")

    def _load_cached_stats(self, cache_path: Path, signature: dict[str, Any]) -> bool:
        if not cache_path.exists():
            return False

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not (isinstance(payload, dict) and payload.get("signature") == signature):
            return False

        self.vertex_mean = payload["vertex_mean"]
        self.vertex_std = payload["vertex_std"]
        self.age_mean = payload["age_mean"]
        self.age_std = payload["age_std"]
        self.bsa_mean = payload["bsa_mean"]
        self.bsa_std = payload["bsa_std"]
        self.heart_rate_mean = payload["heart_rate_mean"]
        self.heart_rate_std = payload["heart_rate_std"]
        return True

    def _compute_and_cache_stats(
        self,
        train_files: list[int],
        cache_path: Path,
        signature: dict[str, Any],
    ) -> None:
        if not train_files:
            raise ValueError(
                "Cannot compute normalization stats because no training files were found."
            )

        stats_files = train_files[: min(self.stats_max_cases, len(train_files))]
        train_vertices: list[torch.Tensor] = []
        train_age: list[torch.Tensor] = []
        train_bsa: list[torch.Tensor] = []
        train_heart_rate: list[torch.Tensor] = []

        with h5py.File(self.h5_data_path, "r") as h5f:
            for item in stats_files:
                x, _, age, bsa, heart_rate, _, _ = self._extract_item_tensors(h5f, item)
                train_vertices.append(x)
                train_age.append(age)
                train_bsa.append(bsa)
                train_heart_rate.append(heart_rate)

        vertices = torch.stack(train_vertices, dim=0)
        age = torch.stack(train_age, dim=0)
        bsa = torch.stack(train_bsa, dim=0)
        heart_rate = torch.stack(train_heart_rate, dim=0)

        batch_size, _, channels = vertices.shape
        vertices = vertices.view(batch_size, -1, TIME_FRAMES, channels)
        vertices = vertices.permute(0, 2, 1, 3).contiguous()
        vertices = vertices.view(-1, vertices.shape[2], channels)

        mean = torch.mean(vertices, dim=0)
        std = torch.std(vertices, dim=0)
        self.vertex_mean = torch.repeat_interleave(mean, TIME_FRAMES, dim=0)
        self.vertex_std = torch.repeat_interleave(std, TIME_FRAMES, dim=0)
        self.age_mean = age.mean()
        self.age_std = age.std()
        self.bsa_mean = bsa.mean()
        self.bsa_std = bsa.std()
        self.heart_rate_mean = heart_rate.mean()
        self.heart_rate_std = heart_rate.std()

        payload = {
            "signature": signature,
            "vertex_mean": self.vertex_mean,
            "vertex_std": self.vertex_std,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "bsa_mean": self.bsa_mean,
            "bsa_std": self.bsa_std,
            "heart_rate_mean": self.heart_rate_mean,
            "heart_rate_std": self.heart_rate_std,
        }
        torch.save(payload, cache_path)

    @staticmethod
    def _extract_item_tensors(h5f: h5py.File, item: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ]:
        pos_ds = MeshDataset._get_dataset(h5f, "pos")
        sex_ds = MeshDataset._get_dataset(h5f, "sex")
        age_ds = MeshDataset._get_dataset(h5f, "age")
        bsa_ds = MeshDataset._get_dataset(h5f, "bsa")
        heart_rate_ds = MeshDataset._get_dataset(h5f, "heart_rate")

        pos = torch.from_numpy(pos_ds[item, ...]).float()
        num_frames, num_nodes = pos.shape[0], pos.shape[1]
        x = pos.permute(1, 0, 2).contiguous().flatten(0, 1)
        sex = torch.tensor(sex_ds[item]).int()
        age = torch.tensor(age_ds[item]).float()
        bsa = torch.tensor(bsa_ds[item]).float()
        heart_rate = torch.tensor(heart_rate_ds[item]).float()
        return x, sex, age, bsa, heart_rate, num_nodes, num_frames

    @staticmethod
    def _get_dataset(h5f: h5py.File, key: str) -> h5py.Dataset:
        if key not in h5f:
            raise ValueError(f"HDF5 dataset must contain '{key}'.")
        obj = h5f[key]
        if not isinstance(obj, h5py.Dataset):
            raise ValueError(f"HDF5 key '{key}' must be a dataset.")
        return obj

    @staticmethod
    def _decode_string(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _standardize(
        cls, value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return (value - mean) / torch.clamp(std, min=EPS)

    def __del__(self) -> None:
        if self.data_handle is not None:
            self.data_handle.close()
            self.data_handle = None

    def len(self) -> int:
        return len(self.file_list)

    def get(self, idx: int) -> Data:
        if self.data_handle is None:
            self.data_handle = h5py.File(self.h5_data_path, "r")

        item = self.file_list[idx]

        x, sex, age, bsa, heart_rate, num_nodes, num_frames = (
            self._extract_item_tensors(self.data_handle, item)
        )
        case_id_ds = self._get_dataset(self.data_handle, "case_id")
        case_id = self._decode_string(case_id_ds[item])

        # Standardize the data
        x = self._standardize(x, self.vertex_mean, self.vertex_std)
        age = self._standardize(age, self.age_mean, self.age_std)
        bsa = self._standardize(bsa, self.bsa_mean, self.bsa_std)
        heart_rate = self._standardize(
            heart_rate, self.heart_rate_mean, self.heart_rate_std
        )

        # Create PyG Data object
        d = Data(
            x=x,
            case_id=case_id,
            sex=sex,
            age=age,
            bsa=bsa,
            heart_rate=heart_rate,
            num_nodes=num_nodes,
            num_frames=num_frames,
        )
        return d
