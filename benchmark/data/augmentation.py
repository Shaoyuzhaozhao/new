"""
物理数据增强模块

实现论文3.2节的三种物理数据增强策略:
1. FDA (频域增强): 智能子载波选择
2. TDA (时域增强): 多尺度时间窗口
3. MDA (运动感知增强): 运动感知随机擦除和平移
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import random

from .spectrogram import (
    compute_motion_statistics,
    select_motion_sensitive_subcarriers,
    detect_motion_regions,
    SpectrogramTransform,
)


class FrequencyDomainAugmentation:
    """
    频域数据增强 (FDA)
    
    利用WiFi OFDM系统的子载波多样性，通过智能子载波选择策略(ISS-N)
    从单个样本生成多个具有不同频率特性的观测
    """
    
    def __init__(
        self,
        num_bands: int = 6,
        motion_threshold: float = 0.5,
        stft_config: Optional[dict] = None,
    ):
        """
        初始化FDA增强器
        
        Args:
            num_bands: 子带数量N，决定生成的增强样本数
            motion_threshold: 运动检测阈值
            stft_config: STFT配置参数
        """
        self.num_bands = num_bands
        self.motion_threshold = motion_threshold
        
        stft_config = stft_config or {}
        self.stft_transform = SpectrogramTransform(**stft_config)
    
    def __call__(
        self,
        csi_data: np.ndarray,
        return_all: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        应用频域增强
        
        Args:
            csi_data: CSI数据，形状可以是 (T,), (T, K), (T, num_links, K), 或更高维
            return_all: 是否返回所有增强样本

        Returns:
            增强后的频谱图或频谱图列表
        """
        # 处理不同维度，转换为2D (T, K)
        if csi_data.ndim == 1:
            # (T,) - 单子载波，无法进行频域增强，直接返回频谱图
            spec = self.stft_transform.transform(csi_data, return_numpy=True)
            return [spec] * self.num_bands if return_all else spec
        elif csi_data.ndim == 2:
            # (T, K) - 已经是正确格式
            csi_2d = csi_data
        elif csi_data.ndim == 3:
            # (T, num_links, K) - 平均链路
            csi_2d = csi_data.mean(axis=1)
        elif csi_data.ndim == 4:
            # (T, Nt, Nr, K) - 平均天线
            csi_2d = csi_data.mean(axis=(1, 2))
        else:
            # 其他情况，展平
            T = csi_data.shape[0]
            csi_2d = csi_data.reshape(T, -1)

        # 选择运动敏感的子载波
        selected_subcarriers = select_motion_sensitive_subcarriers(
            csi_2d, self.num_bands
        )

        # 为每个选中的子载波生成频谱图
        spectrograms = []
        for k in selected_subcarriers:
            if k < csi_2d.shape[1]:
                csi_signal = csi_2d[:, k]
                spec = self.stft_transform.transform(csi_signal, return_numpy=True)
                spectrograms.append(spec)

        if len(spectrograms) == 0:
            # 如果没有选到子载波，使用平均信号
            csi_signal = csi_2d.mean(axis=1)
            spec = self.stft_transform.transform(csi_signal, return_numpy=True)
            spectrograms = [spec]

        if return_all:
            return spectrograms
        else:
            # 随机返回一个
            return random.choice(spectrograms)

    def augment_with_mixing(
        self,
        csi_data: np.ndarray,
        num_groups: int = 3,
    ) -> List[np.ndarray]:
        """
        分组子载波混合策略 (GSM)

        通过k-means聚类将子载波分组，在每组内使用运动统计量
        作为权重进行最大比合并

        Args:
            csi_data: CSI数据
            num_groups: 分组数量

        Returns:
            混合后的频谱图列表
        """
        if csi_data.ndim == 3:
            csi_data = csi_data.mean(axis=1)

        T, K = csi_data.shape

        # 计算每个子载波的运动统计量
        motion_stats = np.array([
            compute_motion_statistics(csi_data[:, k])
            for k in range(K)
        ])

        # 简单分组（均匀划分）
        subcarriers_per_group = K // num_groups
        mixed_spectrograms = []

        for g in range(num_groups):
            start_idx = g * subcarriers_per_group
            end_idx = start_idx + subcarriers_per_group if g < num_groups - 1 else K

            # 获取组内子载波
            group_indices = list(range(start_idx, end_idx))

            # 使用运动敏感度作为权重
            weights = 1 - motion_stats[group_indices]
            weights = weights / (weights.sum() + 1e-8)

            # 加权合并
            mixed_signal = np.zeros(T)
            for i, idx in enumerate(group_indices):
                mixed_signal += weights[i] * csi_data[:, idx]

            # 转换为频谱图
            spec = self.stft_transform.transform(mixed_signal, return_numpy=True)
            mixed_spectrograms.append(spec)

        return mixed_spectrograms


class TemporalDomainAugmentation:
    """
    时域数据增强 (TDA)

    使用多尺度时间窗口进行STFT变换，捕获不同时间尺度的运动特征:
    - 短窗口 (0.5x): 高时间分辨率，捕获快速瞬态事件
    - 标准窗口 (1x): 平衡时频分辨率
    - 长窗口 (2x): 高频率分辨率，精确估计多普勒频移
    """

    def __init__(
        self,
        base_window: int = 256,
        scales: List[float] = [0.5, 1.0, 2.0],
        hop_length: int = 64,
    ):
        """
        初始化TDA增强器

        Args:
            base_window: 基准窗口大小
            scales: 窗口尺度列表
            hop_length: 帧移长度
        """
        self.base_window = base_window
        self.scales = scales
        self.hop_length = hop_length

        # 为每个尺度创建STFT变换器
        self.transforms = {}
        for scale in scales:
            win_length = int(base_window * scale)
            win_length = win_length if win_length % 2 == 0 else win_length + 1
            n_fft = max(win_length, 256)

            self.transforms[scale] = SpectrogramTransform(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            )

    def __call__(
        self,
        csi_signal: np.ndarray,
        return_all: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        应用时域增强

        Args:
            csi_signal: CSI信号，形状为 (T,)
            return_all: 是否返回所有尺度的频谱图

        Returns:
            不同尺度的频谱图或频谱图列表
        """
        spectrograms = []

        for scale in self.scales:
            spec = self.transforms[scale].transform(csi_signal, return_numpy=True)
            spectrograms.append(spec)

        if return_all:
            return spectrograms
        else:
            return random.choice(spectrograms)

    def augment_single(
        self,
        csi_signal: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        使用单一尺度增强

        Args:
            csi_signal: CSI信号
            scale: 指定尺度，None则随机选择

        Returns:
            频谱图
        """
        if scale is None:
            scale = random.choice(self.scales)

        return self.transforms[scale].transform(csi_signal, return_numpy=True)


class MotionAwareAugmentation:
    """
    运动感知数据增强 (MDA)

    基于运动检测结果设计针对性的增强操作:
    - 运动感知随机擦除 (MRE): 选择性擦除边缘运动区域和静态区域
    - 运动感知随机平移 (MRS): 在保持运动核心区域完整的前提下对非运动区域进行循环平移
    """

    def __init__(
        self,
        erase_ratio: float = 0.2,
        static_erase_ratio: float = 0.5,
        energy_threshold: float = 0.7,
        shift_range: float = 0.3,
    ):
        """
        初始化MDA增强器

        Args:
            erase_ratio: 运动区域边缘擦除比例 (20%)
            static_erase_ratio: 静态区域擦除比例 (50%)
            energy_threshold: 运动检测能量阈值
            shift_range: 随机平移范围
        """
        self.erase_ratio = erase_ratio
        self.static_erase_ratio = static_erase_ratio
        self.energy_threshold = energy_threshold
        self.shift_range = shift_range

    def motion_aware_erase(
        self,
        spectrogram: np.ndarray,
        motion_frames: List[int],
        static_frames: List[int],
    ) -> np.ndarray:
        """
        运动感知随机擦除 (MRE)

        - 在运动区间内，只擦除边缘部分(起始和结束的20%)
        - 在非运动时段更激进地擦除(50%)
        - 在频率维度优先擦除能量较低的成分

        Args:
            spectrogram: 频谱图，形状为 (F, T)
            motion_frames: 运动帧索引列表
            static_frames: 静态帧索引列表

        Returns:
            擦除后的频谱图
        """
        spec_aug = spectrogram.copy()
        F, T = spec_aug.shape

        # 处理运动区域 - 只擦除边缘
        if len(motion_frames) > 0:
            motion_frames_sorted = sorted(motion_frames)
            num_edge_frames = max(1, int(len(motion_frames_sorted) * self.erase_ratio))

            # 擦除起始边缘
            for i in range(num_edge_frames):
                if i < len(motion_frames_sorted):
                    frame_idx = motion_frames_sorted[i]
                    # 随机擦除部分频率bin
                    num_erase = random.randint(1, F // 4)
                    erase_start = random.randint(0, F - num_erase)
                    spec_aug[erase_start:erase_start + num_erase, frame_idx] = 0

            # 擦除结束边缘
            for i in range(num_edge_frames):
                idx = len(motion_frames_sorted) - 1 - i
                if idx >= 0:
                    frame_idx = motion_frames_sorted[idx]
                    num_erase = random.randint(1, F // 4)
                    erase_start = random.randint(0, F - num_erase)
                    spec_aug[erase_start:erase_start + num_erase, frame_idx] = 0

        # 处理静态区域 - 更激进地擦除
        if len(static_frames) > 0:
            num_static_erase = int(len(static_frames) * self.static_erase_ratio)
            erase_frames = random.sample(static_frames, min(num_static_erase, len(static_frames)))

            for frame_idx in erase_frames:
                # 擦除能量较低的频率成分
                frame_energy = spec_aug[:, frame_idx]
                low_energy_mask = frame_energy < np.percentile(frame_energy, 50)
                spec_aug[low_energy_mask, frame_idx] = 0

        return spec_aug

    def motion_aware_shift(
        self,
        spectrogram: np.ndarray,
        motion_frames: List[int],
        static_frames: List[int],
    ) -> np.ndarray:
        """
        运动感知随机平移 (MRS)

        在保持运动核心区域完整的前提下对非运动区域进行循环平移

        Args:
            spectrogram: 频谱图，形状为 (F, T)
            motion_frames: 运动帧索引列表
            static_frames: 静态帧索引列表

        Returns:
            平移后的频谱图
        """
        spec_aug = spectrogram.copy()
        F, T = spec_aug.shape

        if len(motion_frames) == 0 or len(static_frames) == 0:
            return spec_aug

        # 找到运动区域的边界
        motion_start = min(motion_frames)
        motion_end = max(motion_frames)

        # 计算平移量
        max_shift = int(min(motion_start, T - motion_end - 1) * self.shift_range)
        if max_shift <= 0:
            return spec_aug

        shift_amount = random.randint(-max_shift, max_shift)

        # 创建平移后的频谱图
        shifted_spec = np.zeros_like(spec_aug)

        for t in range(T):
            if t in motion_frames:
                # 运动帧保持不变
                shifted_spec[:, t] = spec_aug[:, t]
            else:
                # 非运动帧循环平移
                new_t = (t + shift_amount) % T
                # 确保不覆盖运动区域
                if new_t not in motion_frames:
                    shifted_spec[:, new_t] = spec_aug[:, t]
                else:
                    shifted_spec[:, t] = spec_aug[:, t]

        return shifted_spec

    def __call__(
        self,
        spectrogram: np.ndarray,
        apply_erase: bool = True,
        apply_shift: bool = True,
    ) -> np.ndarray:
        """
        应用运动感知增强

        Args:
            spectrogram: 频谱图，形状为 (F, T)
            apply_erase: 是否应用MRE
            apply_shift: 是否应用MRS

        Returns:
            增强后的频谱图
        """
        # 检测运动区域
        motion_frames, static_frames = detect_motion_regions(
            spectrogram, self.energy_threshold
        )

        spec_aug = spectrogram.copy()

        if apply_erase:
            spec_aug = self.motion_aware_erase(spec_aug, motion_frames, static_frames)

        if apply_shift:
            # 重新检测运动区域（因为擦除可能改变了）
            motion_frames, static_frames = detect_motion_regions(
                spec_aug, self.energy_threshold
            )
            spec_aug = self.motion_aware_shift(spec_aug, motion_frames, static_frames)

        return spec_aug


class PhysicalDataAugmentation:
    """
    完整的物理数据增强框架

    整合FDA、TDA、MDA三种策略，提供统一的增强接口
    """

    def __init__(
        self,
        fda_config: Optional[dict] = None,
        tda_config: Optional[dict] = None,
        mda_config: Optional[dict] = None,
        enable_fda: bool = True,
        enable_tda: bool = True,
        enable_mda: bool = True,
    ):
        """
        初始化物理数据增强框架

        Args:
            fda_config: FDA配置
            tda_config: TDA配置
            mda_config: MDA配置
            enable_fda: 是否启用FDA
            enable_tda: 是否启用TDA
            enable_mda: 是否启用MDA
        """
        self.enable_fda = enable_fda
        self.enable_tda = enable_tda
        self.enable_mda = enable_mda

        # 过滤掉 'enabled' 键，只保留实际的配置参数
        def filter_config(cfg):
            if cfg is None:
                return {}
            return {k: v for k, v in cfg.items() if k != 'enabled'}

        if enable_fda:
            self.fda = FrequencyDomainAugmentation(**filter_config(fda_config))

        if enable_tda:
            self.tda = TemporalDomainAugmentation(**filter_config(tda_config))

        if enable_mda:
            self.mda = MotionAwareAugmentation(**filter_config(mda_config))

        # 默认的STFT变换
        self.stft_transform = SpectrogramTransform()

    def augment(
        self,
        csi_data: np.ndarray,
        spectrogram: Optional[np.ndarray] = None,
        return_multiple: bool = False,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        应用物理数据增强

        Args:
            csi_data: 原始CSI数据
            spectrogram: 预计算的频谱图（可选），应该是已归一化的 (F, T) 格式
            return_multiple: 是否返回多个增强样本

        Returns:
            增强后的频谱图或频谱图列表
        """
        augmented_samples = []

        # 如果没有预计算的频谱图，先生成一个
        if spectrogram is None:
            if csi_data.ndim == 1:
                csi_1d = csi_data
            else:
                # 对于 (T, Nt, Nr, K) 格式，平均所有天线和子载波，得到 (T,)
                csi_1d = csi_data.mean(axis=tuple(range(1, csi_data.ndim)))
            spectrogram = self.stft_transform.transform(csi_1d, return_numpy=True)

        # 确保频谱图是2D (F, T)
        original_spec = spectrogram
        if spectrogram.ndim == 3:
            if spectrogram.shape[0] <= spectrogram.shape[1]:
                spectrogram = spectrogram.mean(axis=0)  # (C, F, T) -> (F, T)
            else:
                spectrogram = spectrogram.mean(axis=-1).T  # 转为 (F, T)
        elif spectrogram.ndim > 3:
            spectrogram = spectrogram.reshape(spectrogram.shape[-2], spectrogram.shape[-1])
        elif spectrogram.ndim == 1:
            # 无法处理1D，返回原样
            return original_spec

        # 原始频谱图
        augmented_samples.append(spectrogram.copy())

        # MDA增强 - 直接在传入的频谱图上操作
        if self.enable_mda and spectrogram.ndim == 2:
            mda_spec = self.mda(spectrogram.copy())
            augmented_samples.append(mda_spec)

        # 简化版FDA - 随机频率掩码
        if self.enable_fda and spectrogram.ndim == 2:
            fda_spec = spectrogram.copy()
            F, T = fda_spec.shape
            # 随机掩盖一些频率带
            num_masks = np.random.randint(1, 4)
            for _ in range(num_masks):
                f_start = np.random.randint(0, F - 10)
                f_width = np.random.randint(5, 15)
                fda_spec[f_start:f_start+f_width, :] *= np.random.uniform(0.5, 1.5)
            augmented_samples.append(fda_spec)

        # 简化版TDA - 随机时间掩码
        if self.enable_tda and spectrogram.ndim == 2:
            tda_spec = spectrogram.copy()
            F, T = tda_spec.shape
            # 随机掩盖一些时间帧
            num_masks = np.random.randint(1, 3)
            for _ in range(num_masks):
                t_start = np.random.randint(0, max(1, T - 5))
                t_width = np.random.randint(2, min(8, T - t_start))
                tda_spec[:, t_start:t_start+t_width] *= np.random.uniform(0.5, 1.5)
            augmented_samples.append(tda_spec)

        if return_multiple:
            return augmented_samples
        else:
            # 随机返回一个
            return random.choice(augmented_samples)

    def __call__(
        self,
        csi_data: np.ndarray,
        spectrogram: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        调用增强（返回单个增强样本）
        """
        return self.augment(csi_data, spectrogram, return_multiple=False)


class SpectrogramAugmentationTransform:
    """
    用于PyTorch DataLoader的频谱图增强Transform
    """

    def __init__(
        self,
        enable_fda: bool = True,
        enable_tda: bool = True,
        enable_mda: bool = True,
        p: float = 0.5,
    ):
        """
        Args:
            enable_fda: 是否启用FDA
            enable_tda: 是否启用TDA
            enable_mda: 是否启用MDA
            p: 应用增强的概率
        """
        self.mda = MotionAwareAugmentation() if enable_mda else None
        self.p = p
        self.enable_fda = enable_fda
        self.enable_tda = enable_tda
        self.enable_mda = enable_mda

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        对频谱图应用增强

        Args:
            spectrogram: 频谱图tensor，形状为 (C, F, T) 或 (F, T)

        Returns:
            增强后的频谱图
        """
        if random.random() > self.p:
            return spectrogram

        # 转换为numpy进行增强
        if isinstance(spectrogram, torch.Tensor):
            spec_np = spectrogram.numpy()
            is_tensor = True
        else:
            spec_np = spectrogram
            is_tensor = False

        # 处理通道维度
        if spec_np.ndim == 3:
            # (C, F, T)
            augmented = []
            for c in range(spec_np.shape[0]):
                if self.enable_mda and self.mda is not None:
                    aug_spec = self.mda(spec_np[c])
                else:
                    aug_spec = spec_np[c]
                augmented.append(aug_spec)
            spec_np = np.stack(augmented, axis=0)
        else:
            # (F, T)
            if self.enable_mda and self.mda is not None:
                spec_np = self.mda(spec_np)

        if is_tensor:
            return torch.from_numpy(spec_np).float()
        return spec_np