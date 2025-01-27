"""Advanced causality analysis module for multimodal fusion."""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR

class CausalityAnalyzer:
    """Advanced causality analysis for multimodal signals."""
    
    def __init__(
        self,
        n_bins: int = 20,
        kde_bandwidth: str = 'scott'
    ):
        """Initialize causality analyzer.
        
        Parameters
        ----------
        n_bins : int
            Number of bins for histogram-based methods
        kde_bandwidth : str
            Bandwidth selection method for KDE
        """
        self.n_bins = n_bins
        self.kde_bandwidth = kde_bandwidth
    
    def compute_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        delay: int = 1,
        k_neighbors: int = 3
    ) -> Dict[str, Any]:
        """Compute transfer entropy from source to target.
        
        Parameters
        ----------
        source : np.ndarray
            Source signal
        target : np.ndarray
            Target signal
        delay : int
            Time delay
        k_neighbors : int
            Number of neighbors for density estimation
            
        Returns
        -------
        Dict[str, Any]
            Transfer entropy results
        """
        # Create time-delayed versions
        source_past = source[:-delay]
        target_past = target[:-delay]
        target_present = target[delay:]
        
        # Estimate joint and marginal probabilities using KDE
        joint_data = np.column_stack([target_present, target_past, source_past])
        marginal_data = np.column_stack([target_present, target_past])
        
        kde_joint = KernelDensity(bandwidth=self.kde_bandwidth).fit(joint_data)
        kde_marginal = KernelDensity(bandwidth=self.kde_bandwidth).fit(marginal_data)
        
        # Compute transfer entropy
        te = np.mean(kde_joint.score_samples(joint_data)) - np.mean(kde_marginal.score_samples(marginal_data))
        
        # Compute significance through permutation test
        n_perms = 1000
        null_distribution = np.zeros(n_perms)
        
        for i in range(n_perms):
            perm_source = np.random.permutation(source_past)
            perm_joint = np.column_stack([target_present, target_past, perm_source])
            null_distribution[i] = (
                np.mean(kde_joint.score_samples(perm_joint)) -
                np.mean(kde_marginal.score_samples(marginal_data))
            )
        
        p_value = np.mean(null_distribution >= te)
        
        return {
            'transfer_entropy': te,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'null_distribution': null_distribution
        }
    
    def compute_convergent_cross_mapping(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        embed_dim: int = 3,
        tau: int = 1
    ) -> Dict[str, Any]:
        """Compute convergent cross mapping between signals.
        
        Parameters
        ----------
        signal1 : np.ndarray
            First signal
        signal2 : np.ndarray
            Second signal
        embed_dim : int
            Embedding dimension
        tau : int
            Time delay
            
        Returns
        -------
        Dict[str, Any]
            CCM results
        """
        def create_embedding(x, dim, delay):
            n = len(x) - (dim - 1) * delay
            embedding = np.zeros((n, dim))
            for i in range(dim):
                embedding[:, i] = x[i * delay:i * delay + n]
            return embedding
        
        # Create embeddings
        E1 = create_embedding(signal1, embed_dim, tau)
        E2 = create_embedding(signal2, embed_dim, tau)
        
        # Compute distances and find nearest neighbors
        def find_neighbors(lib, pred, k=embed_dim + 1):
            distances = np.zeros((len(lib), len(pred)))
            for i in range(len(pred)):
                distances[:, i] = np.sqrt(np.sum((lib - pred[i])**2, axis=1))
            return np.argsort(distances, axis=0)[:k, :]
        
        # Cross map in both directions
        neighbors_1to2 = find_neighbors(E1, E2)
        neighbors_2to1 = find_neighbors(E2, E1)
        
        # Compute predictions
        def make_predictions(source, target_embedding, neighbors):
            predictions = np.zeros(len(target_embedding))
            weights = np.exp(-neighbors / embed_dim)
            weights = weights / np.sum(weights, axis=0)
            
            for i in range(len(predictions)):
                predictions[i] = np.sum(source[neighbors[:, i]] * weights[:, i])
            
            return predictions
        
        pred_1to2 = make_predictions(signal2[embed_dim-1:], E1, neighbors_1to2)
        pred_2to1 = make_predictions(signal1[embed_dim-1:], E2, neighbors_2to1)
        
        # Compute correlations
        corr_1to2 = stats.pearsonr(signal2[embed_dim-1:], pred_1to2)[0]
        corr_2to1 = stats.pearsonr(signal1[embed_dim-1:], pred_2to1)[0]
        
        return {
            'ccm_1to2': corr_1to2,
            'ccm_2to1': corr_2to1,
            'asymmetry': abs(corr_1to2 - corr_2to1),
            'predictions': {
                '1to2': pred_1to2,
                '2to1': pred_2to1
            },
            'embeddings': {
                'signal1': E1,
                'signal2': E2
            }
        }
    
    def compute_partial_directed_coherence(
        self,
        signals: List[np.ndarray],
        fs: float = 1.0,
        max_order: int = 10
    ) -> Dict[str, Any]:
        """Compute partial directed coherence for multivariate signals.
        
        Parameters
        ----------
        signals : List[np.ndarray]
            List of signals
        fs : float
            Sampling frequency
        max_order : int
            Maximum VAR model order
            
        Returns
        -------
        Dict[str, Any]
            PDC results
        """
        # Stack signals and fit VAR model
        X = np.column_stack(signals)
        n_signals = len(signals)
        
        # Find optimal model order using AIC
        var_model = VAR(X)
        order_results = []
        
        for p in range(1, max_order + 1):
            try:
                result = var_model.fit(p)
                order_results.append((p, result.aic))
            except:
                continue
        
        if not order_results:
            raise ValueError("Could not fit VAR model")
        
        optimal_order = min(order_results, key=lambda x: x[1])[0]
        model = var_model.fit(optimal_order)
        
        # Compute frequency grid
        freqs = np.linspace(0, fs/2, 128)
        n_freqs = len(freqs)
        
        # Initialize PDC matrix
        pdc = np.zeros((n_signals, n_signals, n_freqs), dtype=complex)
        
        # Get VAR coefficients
        coef = model.coefs.reshape(optimal_order, n_signals, n_signals)
        
        # Compute PDC for each frequency
        for f_idx, freq in enumerate(freqs):
            # Compute transfer function
            H = np.eye(n_signals)
            for p in range(optimal_order):
                H -= coef[p] * np.exp(-2j * np.pi * freq * (p + 1) / fs)
            
            # Compute PDC
            for i in range(n_signals):
                for j in range(n_signals):
                    num = abs(H[i, j])
                    den = np.sqrt(np.sum(abs(H[:, j])**2))
                    pdc[i, j, f_idx] = num / den if den != 0 else 0
        
        return {
            'pdc': pdc,
            'frequencies': freqs,
            'var_order': optimal_order,
            'var_model': model,
            'aic': dict(order_results)
        }
    
    def compute_information_flow(
        self,
        source: np.ndarray,
        target: np.ndarray,
        conditional: Optional[np.ndarray] = None,
        k_neighbors: int = 3
    ) -> Dict[str, Any]:
        """Compute conditional mutual information and information flow.
        
        Parameters
        ----------
        source : np.ndarray
            Source signal
        target : np.ndarray
            Target signal
        conditional : Optional[np.ndarray]
            Conditional signal
        k_neighbors : int
            Number of neighbors for MI estimation
            
        Returns
        -------
        Dict[str, Any]
            Information flow results
        """
        # Compute mutual information
        mi = mutual_info_regression(
            source.reshape(-1, 1),
            target,
            n_neighbors=k_neighbors
        )[0]
        
        # Compute conditional mutual information if provided
        cmi = None
        if conditional is not None:
            # Stack source and conditional
            X = np.column_stack([source, conditional])
            cmi = mutual_info_regression(
                X,
                target,
                n_neighbors=k_neighbors
            )[0]
        
        # Compute time-delayed mutual information
        delays = np.arange(1, 11)
        delayed_mi = []
        
        for delay in delays:
            delayed_source = source[:-delay]
            delayed_target = target[delay:]
            delayed_mi.append(
                mutual_info_regression(
                    delayed_source.reshape(-1, 1),
                    delayed_target,
                    n_neighbors=k_neighbors
                )[0]
            )
        
        return {
            'mutual_information': mi,
            'conditional_mutual_information': cmi,
            'time_delayed_mi': {
                'delays': delays,
                'values': delayed_mi
            },
            'information_flow': mi - (cmi if cmi is not None else 0)
        } 