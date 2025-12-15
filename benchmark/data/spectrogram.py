"""
时频分析模块

提供CSI数据到频谱图的转换功能
"""

import numpy as np
import torch
from scipy import signal
from typing import Optional, Union, Tuple


class SpectrogramTransform:
    """
    STFT频谱图变换

    将1D时域信号转换为2D时频表示
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        """
        初始化STFT变换器

        Args:
            n_fft: FFT点数
            hop_length: 帧移长度
            win_length: 窗口长度
            window: 窗口类型
            center: 是否中心化
            pad_mode: 填充模式
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # 预计算窗函数
        self.window_fn = signal.get_window(window, self.win_length)

    def transform(
        self,
        x: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        执行STFT变换

        Args:
            x: 输入信号，形状为 (T,) 或 (T, K)
            return_numpy: 是否返回numpy数组

        Returns:
            频谱图，形状为 (F, T')
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        # 如果是2D输入 (T, K)，对每个子载波做STFT然后平均
        if x.ndim == 2:
            T, K = x.shape
            spectrograms = []
            for k in range(K):
                f, t, Zxx = signal.stft(
                    x[:, k],
                    nperseg=self.win_length,
                    noverlap=self.win_length - self.hop_length,
                    nfft=self.n_fft,
                    window=self.window,
                )
                spectrograms.append(np.abs(Zxx))
            # 平均所有子载波的频谱图
            spectrogram = np.mean(spectrograms, axis=0)
        else:
            # 1D输入
            f, t, Zxx = signal.stft(
                x,
                nperseg=self.win_length,
                noverlap=self.win_length - self.hop_length,
                nfft=self.n_fft,
                window=self.window,
            )
            spectrogram = np.abs(Zxx)

        if return_numpy:
            return spectrogram
        return torch.from_numpy(spectrogram).float()

    def inverse_transform(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        length: Optional[int] = None,
    ) -> np.ndarray:
        """
        执行逆STFT变换（仅用于幅度谱，相位使用Griffin-Lim）

        Args:
            spectrogram: 频谱图
            length: 输出信号长度

        Returns:
            重建的时域信号
        """
        if isinstance(spectrogram, torch.Tensor):
            spectrogram = spectrogram.numpy()

        # Griffin-Lim算法进行相位恢复
        n_iter = 32
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

        for _ in range(n_iter):
            _, x_reconstructed = signal.istft(
                spectrogram * angles,
                nperseg=self.win_length,
                noverlap=self.win_length - self.hop_length,
                nfft=self.n_fft,
                window=self.window,
            )
            _, _, stft_matrix = signal.stft(
                x_reconstructed,
                nperseg=self.win_length,
                noverlap=self.win_length - self.hop_length,
                nfft=self.n_fft,
                window=self.window,
            )
            angles = np.exp(1j * np.angle(stft_matrix))

        _, x_final = signal.istft(
            spectrogram * angles,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            window=self.window,
        )

        if length is not None:
            x_final = x_final[:length]

        return x_final


class CSIToSpectrogram:
    """
    完整的CSI到频谱图转换管道

    包括：
    1. CSI幅度/相位提取
    2. STFT变换
    3. 频谱图归一化
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        normalize: bool = True,
        log_scale: bool = True,
        use_amplitude: bool = True,
    ):
        """
        初始化转换管道

        Args:
            n_fft: FFT点数
            hop_length: 帧移长度
            normalize: 是否归一化
            log_scale: 是否使用对数刻度
            use_amplitude: 是否使用幅度(True)或使用复数(False)
        """
        self.stft_transform = SpectrogramTransform(
            n_fft=n_fft,
            hop_length=hop_length,
        )
        self.normalize = normalize
        self.log_scale = log_scale
        self.use_amplitude = use_amplitude

    def __call__(
        self,
        csi_data: Union[np.ndarray, torch.Tensor],
        subcarrier_idx: Optional[int] = None,
        link_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        将CSI数据转换为频谱图

        Args:
            csi_data: CSI数据，形状可以是:
                - (T, Nt, Nr, K): 完整CSI张量
                - (T, num_links, K): 简化的CSI张量
                - (T, K): 单链路CSI
                - (T,): 单子载波CSI
            subcarrier_idx: 指定子载波索引
            link_idx: 指定链路索引

        Returns:
            频谱图tensor，形状为 (F, T')
        """
        # 转换为numpy进行处理
        if isinstance(csi_data, torch.Tensor):
            csi_data = csi_data.numpy()

        # 处理不同维度的CSI数据
        if csi_data.ndim == 4:
            # (T, Nt, Nr, K) -> 平均天线，保留子载波 -> (T, K)
            T, Nt, Nr, K = csi_data.shape
            if link_idx is not None and subcarrier_idx is not None:
                nt_idx = link_idx // Nr
                nr_idx = link_idx % Nr
                csi_signal = csi_data[:, nt_idx, nr_idx, subcarrier_idx]  # (T,)
            elif subcarrier_idx is not None:
                # 平均所有链路，保留指定子载波
                csi_signal = csi_data[:, :, :, subcarrier_idx].mean(axis=(1, 2))  # (T,)
            else:
                # 平均所有天线，保留所有子载波 -> (T, K)
                csi_signal = csi_data.mean(axis=(1, 2))  # (T, K)

        elif csi_data.ndim == 3:
            # (T, num_links, K)
            if subcarrier_idx is not None:
                csi_signal = csi_data[:, :, subcarrier_idx].mean(axis=1)  # (T,)
            else:
                # 平均链路，保留子载波 -> (T, K)
                csi_signal = csi_data.mean(axis=1)  # (T, K)

        elif csi_data.ndim == 2:
            # (T, K) - 已经是正确形状
            if subcarrier_idx is not None:
                csi_signal = csi_data[:, subcarrier_idx]  # (T,)
            else:
                csi_signal = csi_data  # (T, K)
        else:
            # (T,)
            csi_signal = csi_data

        # STFT变换 - 现在支持 (T, K) 输入
        spectrogram = self.stft_transform.transform(csi_signal, return_numpy=False)

        # 对数刻度
        if self.log_scale:
            spectrogram = torch.log1p(spectrogram)

        # 归一化
        if self.normalize:
            mean = spectrogram.mean()
            std = spectrogram.std()
            if std > 0:
                spectrogram = (spectrogram - mean) / std

        return spectrogram


class MultiScaleSpectrogram:
    """
    多尺度频谱图生成

    使用不同的窗口大小生成多个频谱图
    """

    def __init__(
        self,
        n_ffts: Tuple[int, ...] = (128, 256, 512),
        hop_length: int = 64,
        normalize: bool = True,
    ):
        """
        初始化多尺度频谱图生成器

        Args:
            n_ffts: 多个FFT点数
            hop_length: 帧移长度
            normalize: 是否归一化
        """
        self.transforms = [
            CSIToSpectrogram(n_fft=n_fft, hop_length=hop_length, normalize=normalize)
            for n_fft in n_ffts
        ]

    def __call__(self, csi_data: np.ndarray) -> list:
        """
        生成多尺度频谱图

        Args:
            csi_data: CSI数据

        Returns:
            多尺度频谱图列表
        """
        return [transform(csi_data) for transform in self.transforms]