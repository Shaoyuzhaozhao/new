"""
数据集加载模块

支持WiMANS数据集的加载、预处理和数据增强
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Callable
import scipy.io as scio
from sklearn.model_selection import train_test_split

from .spectrogram import CSIToSpectrogram, SpectrogramTransform
from .augmentation import PhysicalDataAugmentation, SpectrogramAugmentationTransform


class WiMANSDataset(Dataset):
    """
    WiMANS多用户WiFi感知数据集
    """

    # 固定的输出尺寸 (频率bins, 时间帧)
    FIXED_FREQ_BINS = 129
    FIXED_TIME_FRAMES = 47

    def __init__(
        self,
        data_dir: str,
        annotation_path: str,
        task: str = "activity",
        num_users: List[str] = ["5"],
        wifi_band: List[str] = ["2.4"],
        environment: List[str] = ["classroom", "meeting_room", "empty_room"],
        use_spectrogram: bool = True,
        stft_config: Optional[dict] = None,
        augmentation: Optional[PhysicalDataAugmentation] = None,
        transform: Optional[Callable] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        random_seed: int = 42,
        use_amp: bool = True,
    ):
        self.data_dir = data_dir
        self.task = task
        self.use_spectrogram = use_spectrogram
        self.augmentation = augmentation
        self.transform = transform
        self.split = split

        # 自动检测数据类型
        self.use_amp = self._detect_data_type(data_dir)

        # 加载标注
        self.annotations = pd.read_csv(annotation_path, dtype=str)

        # 筛选数据
        self.annotations = self._filter_data(
            self.annotations, num_users, wifi_band, environment
        )

        # 划分训练/测试集
        if split != "all":
            self.annotations = self._split_data(
                self.annotations, split, train_ratio, random_seed
            )

        # 初始化频谱图转换器
        if use_spectrogram:
            stft_config = stft_config or {}
            self.spectrogram_transform = CSIToSpectrogram(**stft_config)

        # 构建标签映射
        self._build_label_mapping()

        print(f"Activity classes found: {self.label_map}")
        print(f"Loaded {len(self.annotations)} samples for {split} set")

    def _detect_data_type(self, data_dir: str) -> bool:
        """自动检测数据类型"""
        if not os.path.exists(data_dir):
            return True

        files = os.listdir(data_dir)
        npy_files = [f for f in files if f.endswith('.npy')]
        mat_files = [f for f in files if f.endswith('.mat')]

        if len(npy_files) > 0:
            print(f"Detected .npy files in {data_dir}, using amplitude data")
            return True
        elif len(mat_files) > 0:
            print(f"Detected .mat files in {data_dir}, using raw CSI data")
            return False
        else:
            return True

    def _filter_data(
        self,
        df: pd.DataFrame,
        num_users: List[str],
        wifi_band: List[str],
        environment: List[str],
    ) -> pd.DataFrame:
        """筛选数据"""
        # 兼容不同列名
        num_users_col = "number_of_users" if "number_of_users" in df.columns else "num_users"
        if num_users_col in df.columns:
            df = df[df[num_users_col].isin(num_users)]

        if "wifi_band" in df.columns:
            df = df[df["wifi_band"].isin(wifi_band)]

        if "environment" in df.columns:
            df = df[df["environment"].isin(environment)]

        return df.reset_index(drop=True)

    def _split_data(
        self,
        df: pd.DataFrame,
        split: str,
        train_ratio: float,
        random_seed: int,
    ) -> pd.DataFrame:
        """划分训练/测试集"""
        sample_id_col = None
        for col_name in ["label", "sample_id", "name", "id", "filename"]:
            if col_name in df.columns:
                sample_id_col = col_name
                break

        if sample_id_col:
            sample_ids = df[sample_id_col].unique()
        else:
            sample_ids = df.index.values

        train_ids, test_ids = train_test_split(
            sample_ids,
            train_size=train_ratio,
            random_state=random_seed,
        )

        if split == "train":
            if sample_id_col:
                return df[df[sample_id_col].isin(train_ids)].reset_index(drop=True)
            else:
                return df.iloc[list(train_ids)].reset_index(drop=True)
        else:
            if sample_id_col:
                return df[df[sample_id_col].isin(test_ids)].reset_index(drop=True)
            else:
                return df.iloc[list(test_ids)].reset_index(drop=True)

    def _build_label_mapping(self):
        """构建标签映射"""
        if self.task == "activity":
            # WiMANS格式
            if "user_1_activity" in self.annotations.columns:
                self.label_col = "user_1_activity"
            elif "activity" in self.annotations.columns:
                self.label_col = "activity"
            else:
                self.label_col = "label"

            self.num_classes = 9
            valid_labels = self.annotations[self.label_col].dropna().unique()
            unique_labels = sorted(valid_labels)
            self.label_map = {label: i for i, label in enumerate(unique_labels)}

        elif self.task == "identity":
            self.label_col = "user_id" if "user_id" in self.annotations.columns else "identity"
            unique_labels = self.annotations[self.label_col].unique()
            self.label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
            self.num_classes = len(self.label_map)
        elif self.task == "location":
            if "user_1_location" in self.annotations.columns:
                self.label_col = "user_1_location"
            else:
                self.label_col = "location"
            unique_labels = self.annotations[self.label_col].dropna().unique()
            self.label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
            self.num_classes = len(self.label_map)

    def _load_csi(self, sample_name: str) -> np.ndarray:
        """加载CSI数据"""
        if self.use_amp:
            file_path = os.path.join(self.data_dir, f"{sample_name}.npy")
            if os.path.exists(file_path):
                return np.load(file_path)

        mat_path = os.path.join(self.data_dir, f"{sample_name}.mat")
        if os.path.exists(mat_path):
            return self._load_mat_csi(mat_path)

        raise FileNotFoundError(f"CSI data not found for {sample_name}")

    def _load_mat_csi(self, mat_path: str) -> np.ndarray:
        """
        加载.mat格式的原始CSI数据

        保留完整的天线维度，与.npy格式一致

        Returns:
            CSI幅度数据，形状为 (T, 3, 3, 30)
        """
        mat_data = scio.loadmat(mat_path)

        if 'trace' not in mat_data:
            raise ValueError(f"No 'trace' key found in {mat_path}")

        trace = mat_data['trace']
        num_packets = trace.shape[0]

        csi_list = []
        for i in range(num_packets):
            try:
                packet_wrapper = trace[i, 0]
                csi = packet_wrapper['csi'][0, 0]

                if isinstance(csi, np.ndarray) and np.iscomplexobj(csi):
                    # 保留完整形状 (3, 3, 30)，不做平均
                    csi_amp = np.abs(csi)
                    csi_list.append(csi_amp)

            except Exception as e:
                continue

        if len(csi_list) > 0:
            # 堆叠得到 (T, 3, 3, 30)
            csi_data = np.stack(csi_list, axis=0)
            return csi_data.astype(np.float32)
        else:
            raise ValueError(f"No valid CSI packets found in {mat_path}")

    def _fix_spectrogram_size(self, spectrogram: np.ndarray) -> np.ndarray:
        """将频谱图调整为固定尺寸"""
        if spectrogram.ndim == 2:
            f, t = spectrogram.shape
            target_f, target_t = self.FIXED_FREQ_BINS, self.FIXED_TIME_FRAMES

            result = np.zeros((target_f, target_t), dtype=spectrogram.dtype)
            copy_f = min(f, target_f)
            copy_t = min(t, target_t)
            result[:copy_f, :copy_t] = spectrogram[:copy_f, :copy_t]

            return result

        elif spectrogram.ndim == 3:
            c, f, t = spectrogram.shape
            target_f, target_t = self.FIXED_FREQ_BINS, self.FIXED_TIME_FRAMES

            result = np.zeros((c, target_f, target_t), dtype=spectrogram.dtype)
            copy_f = min(f, target_f)
            copy_t = min(t, target_t)
            result[:, :copy_f, :copy_t] = spectrogram[:, :copy_f, :copy_t]

            return result

        return spectrogram

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        row = self.annotations.iloc[idx]

        # 获取样本名称
        for col_name in ["label", "sample_id", "name", "id", "filename"]:
            if col_name in row.index:
                sample_name = str(row[col_name])
                break
        else:
            sample_name = str(row.name) if isinstance(row.name, str) else f"act_{idx}"

        # 加载CSI数据
        csi_data = self._load_csi(sample_name)

        # 转换为频谱图
        if self.use_spectrogram:
            spectrogram = self.spectrogram_transform(csi_data)
            if isinstance(spectrogram, torch.Tensor):
                spectrogram = spectrogram.numpy()
        else:
            spectrogram = csi_data

        # 调整为固定尺寸
        spectrogram = self._fix_spectrogram_size(spectrogram)

        # 应用数据增强
        if self.augmentation is not None and self.split == "train":
            spectrogram = self.augmentation(csi_data, spectrogram)
            if isinstance(spectrogram, torch.Tensor):
                spectrogram = spectrogram.numpy()
            spectrogram = self._fix_spectrogram_size(spectrogram)

        # 转为tensor
        spectrogram = torch.from_numpy(spectrogram).float()

        # 确保有通道维度
        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)

        # 应用额外变换
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        # 获取标签
        label_str = str(row[self.label_col])
        label = self.label_map.get(label_str, -1)

        if label < 0 or label >= self.num_classes:
            label = 0

        return spectrogram, label


class WiMANSSpectrogramDataset(Dataset):
    """预计算频谱图的WiMANS数据集"""

    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        augmentation: Optional[SpectrogramAugmentationTransform] = None,
        normalize: bool = True,
    ):
        self.spectrograms = spectrograms
        self.labels = labels
        self.augmentation = augmentation
        self.normalize = normalize

        if normalize:
            self.mean = np.mean(spectrograms)
            self.std = np.std(spectrograms) + 1e-8

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spectrogram = self.spectrograms[idx].copy()
        label = int(self.labels[idx])

        if self.normalize:
            spectrogram = (spectrogram - self.mean) / self.std

        spectrogram = torch.from_numpy(spectrogram).float()

        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)

        if self.augmentation is not None:
            spectrogram = self.augmentation(spectrogram)

        return spectrogram, label


def create_data_loaders(
    data_dir: str,
    annotation_path: str,
    task: str = "activity",
    num_users: List[str] = ["5"],
    wifi_band: List[str] = ["2.4"],
    environment: List[str] = ["classroom", "meeting_room", "empty_room"],
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_config: Optional[dict] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和测试数据加载器"""
    augmentation = None
    if augmentation_config is not None and augmentation_config.get("enabled", True):
        augmentation = PhysicalDataAugmentation(
            fda_config=augmentation_config.get("fda", {}),
            tda_config=augmentation_config.get("tda", {}),
            mda_config=augmentation_config.get("mda", {}),
            enable_fda=augmentation_config.get("fda", {}).get("enabled", True),
            enable_tda=augmentation_config.get("tda", {}).get("enabled", True),
            enable_mda=augmentation_config.get("mda", {}).get("enabled", True),
        )

    train_dataset = WiMANSDataset(
        data_dir=data_dir,
        annotation_path=annotation_path,
        task=task,
        num_users=num_users,
        wifi_band=wifi_band,
        environment=environment,
        augmentation=augmentation,
        split="train",
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    test_dataset = WiMANSDataset(
        data_dir=data_dir,
        annotation_path=annotation_path,
        task=task,
        num_users=num_users,
        wifi_band=wifi_band,
        environment=environment,
        augmentation=None,
        split="test",
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def create_cross_environment_loaders(
    data_dir: str,
    annotation_path: str,
    train_env: str,
    test_env: str,
    task: str = "activity",
    num_users: List[str] = ["5"],
    wifi_band: List[str] = ["2.4"],
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_config: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader]:
    """创建跨环境训练/测试数据加载器"""
    augmentation = None
    if augmentation_config is not None:
        augmentation = PhysicalDataAugmentation(
            fda_config=augmentation_config.get("fda", {}),
            tda_config=augmentation_config.get("tda", {}),
            mda_config=augmentation_config.get("mda", {}),
        )

    train_dataset = WiMANSDataset(
        data_dir=data_dir,
        annotation_path=annotation_path,
        task=task,
        num_users=num_users,
        wifi_band=wifi_band,
        environment=[train_env],
        augmentation=augmentation,
        split="all",
    )

    test_dataset = WiMANSDataset(
        data_dir=data_dir,
        annotation_path=annotation_path,
        task=task,
        num_users=num_users,
        wifi_band=wifi_band,
        environment=[test_env],
        augmentation=None,
        split="all",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader