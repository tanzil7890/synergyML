"""Advanced wavelet analysis module for multimodal fusion."""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pywt
from scipy import signal
import matplotlib.pyplot as plt

class WaveletAnalyzer:
    """Advanced wavelet analysis for multimodal signals."""
    
    def __init__(
        self,
        default_wavelet: str = 'cmor1.5-1.0',
        n_scales: int = 64
    ):
        """Initialize wavelet analyzer.
        
        Parameters
        ----------
        default_wavelet : str
            Default wavelet to use
        n_scales : int
            Number of scales for analysis
        """
        self.default_wavelet = default_wavelet
        self.n_scales = n_scales
    
    def compute_wavelet_coherence(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        fs: float = 1.0,
        freq_range: Optional[Tuple[float, float]] = None,
        smoothing_radius: int = 5
    ) -> Dict[str, Any]:
        """Compute wavelet coherence between two signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
        fs : float
            Sampling frequency
        freq_range : Optional[Tuple[float, float]]
            Frequency range to analyze
        smoothing_radius : int
            Radius for coherence smoothing
            
        Returns
        -------
        Dict[str, Any]
            Wavelet coherence results
        """
        # Normalize signals
        s1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
        s2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
        
        # Set up scales
        if freq_range is not None:
            min_freq, max_freq = freq_range
            scales = pywt.frequency2scale(
                self.default_wavelet,
                np.logspace(
                    np.log10(min_freq),
                    np.log10(max_freq),
                    num=self.n_scales
                ),
                fs
            )
        else:
            scales = np.arange(1, self.n_scales + 1)
        
        # Compute CWT for both signals
        coef1, freqs = pywt.cwt(s1_norm, scales, self.default_wavelet, 1/fs)
        coef2, _ = pywt.cwt(s2_norm, scales, self.default_wavelet, 1/fs)
        
        # Compute cross-wavelet transform
        Wxy = coef1 * np.conj(coef2)
        
        # Smooth the cross-wavelet spectrum
        kernel = np.ones((smoothing_radius, smoothing_radius)) / (smoothing_radius**2)
        Wxy_smooth = signal.convolve2d(np.abs(Wxy)**2, kernel, mode='same')
        
        # Compute auto-spectra
        Wxx = np.abs(coef1)**2
        Wyy = np.abs(coef2)**2
        
        # Smooth auto-spectra
        Wxx_smooth = signal.convolve2d(Wxx, kernel, mode='same')
        Wyy_smooth = signal.convolve2d(Wyy, kernel, mode='same')
        
        # Compute wavelet coherence
        coherence = np.abs(Wxy_smooth) / np.sqrt(Wxx_smooth * Wyy_smooth)
        
        # Compute phase difference
        phase_diff = np.angle(Wxy)
        
        return {
            'coherence': coherence,
            'phase_difference': phase_diff,
            'frequencies': freqs,
            'scales': scales,
            'cross_spectrum': Wxy,
            'time_frequency_coherence': {
                'magnitude': coherence,
                'phase': phase_diff
            }
        }
    
    def compute_wavelet_synchrosqueezed(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        freq_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Compute synchrosqueezed wavelet transform.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        fs : float
            Sampling frequency
        freq_range : Optional[Tuple[float, float]]
            Frequency range to analyze
            
        Returns
        -------
        Dict[str, Any]
            Synchrosqueezed transform results
        """
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / np.std(signal)
        
        # Set up frequency bins
        if freq_range is not None:
            min_freq, max_freq = freq_range
            freqs = np.logspace(
                np.log10(min_freq),
                np.log10(max_freq),
                num=self.n_scales
            )
        else:
            freqs = np.linspace(1, fs/2, self.n_scales)
        
        # Compute CWT
        coef, frequencies = pywt.cwt(
            signal_norm,
            np.arange(1, self.n_scales + 1),
            self.default_wavelet,
            1/fs
        )
        
        # Compute instantaneous frequencies
        dt = 1/fs
        dcoef = np.gradient(coef, dt, axis=1)
        inst_freqs = np.imag(dcoef / coef) / (2 * np.pi)
        
        # Perform synchrosqueezing
        sst = np.zeros((len(freqs), len(signal)), dtype=complex)
        
        for i, freq in enumerate(freqs):
            mask = np.abs(inst_freqs - freq) < (freq * 0.1)  # 10% frequency tolerance
            sst[i, :] = np.sum(coef * mask, axis=0)
        
        return {
            'transform': sst,
            'frequencies': freqs,
            'instantaneous_frequencies': inst_freqs,
            'original_cwt': coef,
            'original_frequencies': frequencies
        }
    
    def compute_ridge_extraction(
        self,
        wavelet_transform: np.ndarray,
        frequencies: np.ndarray,
        penalty: float = 0.5
    ) -> Dict[str, Any]:
        """Extract ridges from wavelet transform.
        
        Parameters
        ----------
        wavelet_transform : np.ndarray
            Wavelet transform coefficients
        frequencies : np.ndarray
            Frequency array
        penalty : float
            Ridge extraction penalty factor
            
        Returns
        -------
        Dict[str, Any]
            Ridge extraction results
        """
        power = np.abs(wavelet_transform)**2
        n_freqs, n_times = power.shape
        
        # Initialize ridge arrays
        ridge_freqs = np.zeros(n_times)
        ridge_indices = np.zeros(n_times, dtype=int)
        
        # Extract ridge for each time point
        for t in range(n_times):
            # Find local maxima in frequency
            peaks = signal.find_peaks(power[:, t])[0]
            
            if len(peaks) > 0:
                # Select strongest peak
                peak_idx = peaks[np.argmax(power[peaks, t])]
                ridge_indices[t] = peak_idx
                ridge_freqs[t] = frequencies[peak_idx]
        
        # Compute ridge properties
        ridge_amplitude = np.abs(wavelet_transform[ridge_indices, np.arange(n_times)])
        ridge_phase = np.angle(wavelet_transform[ridge_indices, np.arange(n_times)])
        
        return {
            'frequencies': ridge_freqs,
            'indices': ridge_indices,
            'amplitude': ridge_amplitude,
            'phase': ridge_phase,
            'significance': ridge_amplitude / np.mean(ridge_amplitude)
        }
    
    def compute_wavelet_bispectrum(
        self,
        signal: np.ndarray,
        fs: float = 1.0,
        freq_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """Compute wavelet bispectrum for nonlinear interactions.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        fs : float
            Sampling frequency
        freq_range : Optional[Tuple[float, float]]
            Frequency range to analyze
            
        Returns
        -------
        Dict[str, Any]
            Bispectrum analysis results
        """
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / np.std(signal)
        
        # Compute CWT
        coef, freqs = pywt.cwt(
            signal_norm,
            np.arange(1, self.n_scales + 1),
            self.default_wavelet,
            1/fs
        )
        
        # Initialize bispectrum matrix
        n_freqs = len(freqs)
        bispectrum = np.zeros((n_freqs, n_freqs), dtype=complex)
        
        # Compute bispectrum
        for i in range(n_freqs):
            for j in range(n_freqs):
                if i + j < n_freqs:
                    # Compute triple product for bispectrum
                    bispectrum[i, j] = np.mean(
                        coef[i, :] * coef[j, :] * np.conj(coef[i+j, :])
                    )
        
        # Compute bicoherence
        bicoherence = np.abs(bispectrum) / np.sqrt(
            np.outer(
                np.mean(np.abs(coef)**2, axis=1),
                np.mean(np.abs(coef)**2, axis=1)
            )
        )
        
        return {
            'bispectrum': bispectrum,
            'bicoherence': bicoherence,
            'frequencies': freqs,
            'nonlinear_coupling': {
                'magnitude': np.abs(bispectrum),
                'phase': np.angle(bispectrum)
            }
        } 