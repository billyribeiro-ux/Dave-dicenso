"""
evolution/regime_detector.py — Market regime detection via latent space analysis.

FULL implementation:
- Latent space clustering with GaussianMixture and DBSCAN
- Regime labeling, tracking, and persistence
- Regime transition probability matrix
- Real-time alert when a new regime is detected
- Cosine distance sliding window for boundary detection
- Regime fingerprinting for future recognition

No volatility indicators, no trend filters — purely latent-space-driven.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from utils.logger import get_logger


class RegimeDetector:
    """Detects market regime changes through latent space drift.

    Uses multiple clustering approaches:
    - GaussianMixture for soft regime assignment
    - DBSCAN for outlier/anomaly detection
    - Cosine distance sliding window for boundary detection

    Maintains a transition probability matrix to predict regime changes.
    """

    DEFAULT_THRESHOLD = 0.5
    MIN_REGIME_DURATION = 20
    SLIDING_WINDOW_SIZE = 10

    def __init__(self, world_model=None, threshold: float = DEFAULT_THRESHOLD):
        self.world_model = world_model
        self.threshold = threshold
        self.logger = get_logger()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Regime tracking
        self.regime_boundaries: List[dict] = []
        self.current_regime_id: int = 0
        self.latent_history: List[np.ndarray] = []

        # Clustering models (fit on accumulated data)
        self._gmm: Optional[GaussianMixture] = None
        self._dbscan: Optional[DBSCAN] = None
        self._gmm_labels: Optional[np.ndarray] = None
        self._dbscan_labels: Optional[np.ndarray] = None
        self._n_regimes: int = 0

        # Transition matrix
        self._transition_counts: Optional[np.ndarray] = None
        self._transition_matrix: Optional[np.ndarray] = None
        self._regime_sequence: List[int] = []

        # Alert callback
        self._alert_callback: Optional[callable] = None

    # ======================================================================
    # detect_regimes — Main scanning method
    # ======================================================================

    def detect_regimes(
        self, prices: np.ndarray, window_size: int = 500, stride: int = 50
    ) -> List[dict]:
        """Scan through price data and detect regime boundaries.

        Uses a sliding window of cosine distances: compares the
        mean of the last SLIDING_WINDOW_SIZE latent vectors against
        the current one. If the distance exceeds the threshold,
        a regime boundary is marked.

        Args:
            prices: 1D numpy array of close prices.
            window_size: Number of ticks per encoding window.
            stride: Steps between consecutive windows.

        Returns:
            List of regime boundary dicts.
        """
        if self.world_model is None:
            self.logger.error("World model not set for regime detection")
            return []

        if len(prices) < window_size + stride:
            self.logger.warning("Not enough data for regime detection")
            return []

        self.world_model.eval()
        self.world_model.to(self.device)

        num_windows = (len(prices) - window_size) // stride

        self.logger.info(
            f"Detecting regimes: {num_windows} windows, "
            f"stride={stride}, threshold={self.threshold}"
        )

        # Reset state
        self.latent_history = []
        self.regime_boundaries = []
        self._regime_sequence = []
        latent_vectors = []
        boundaries = []
        current_regime_start = 0
        regime_id = 0

        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window = prices[start:end].astype(np.float32)

            latent = self._encode_price_window(window)
            if latent is None:
                continue

            latent_vectors.append(latent)
            self.latent_history.append(latent)
            self._regime_sequence.append(regime_id)

            # Sliding window cosine distance comparison
            if len(latent_vectors) >= self.SLIDING_WINDOW_SIZE:
                prev_mean = np.mean(
                    latent_vectors[-self.SLIDING_WINDOW_SIZE - 1:-1], axis=0
                )
                distance = cosine(prev_mean, latent)

                if distance > self.threshold:
                    boundary = {
                        "index": start,
                        "window_idx": i,
                        "regime_id": regime_id,
                        "distance": float(distance),
                        "previous_regime_length": start - current_regime_start,
                        "timestamp": datetime.now().isoformat(),
                    }
                    boundaries.append(boundary)

                    self.logger.info(
                        f"Regime boundary at tick {start}: "
                        f"cosine distance = {distance:.4f}, "
                        f"prev regime length = {start - current_regime_start}"
                    )

                    # Alert callback
                    self._alert_regime_change(regime_id, regime_id + 1, distance)

                    regime_id += 1
                    current_regime_start = start
                    self._regime_sequence.append(regime_id)

        self.regime_boundaries = boundaries
        self.current_regime_id = regime_id

        self.logger.info(f"Detected {len(boundaries)} regime boundaries, {regime_id + 1} total regimes")
        return boundaries

    # ======================================================================
    # cluster_latent_vectors — GaussianMixture + DBSCAN
    # ======================================================================

    def cluster_latent_vectors(
        self, n_clusters: Optional[int] = None, method: str = "gmm"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Cluster latent vectors using GaussianMixture or DBSCAN.

        GaussianMixture provides soft assignments (probabilities per regime).
        DBSCAN identifies outliers that don't belong to any regime.

        Args:
            n_clusters: Number of clusters. Auto-detected if None.
            method: "gmm" (GaussianMixture) or "dbscan" or "kmeans" or "both".

        Returns:
            (labels, cluster_centers_or_None)
            For GMM: labels are hard assignments, centers are means.
            For DBSCAN: labels include -1 for noise points.
        """
        if len(self.latent_history) < 5:
            self.logger.warning("Not enough latent vectors for clustering")
            return np.array([]), None

        vectors = np.stack(self.latent_history)

        if n_clusters is None:
            n_clusters = max(2, min(10, int(np.sqrt(len(vectors)))))

        labels = np.zeros(len(vectors), dtype=int)
        centers = None

        if method in ("gmm", "both"):
            self._gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                random_state=42,
                n_init=3,
                max_iter=200,
            )
            self._gmm_labels = self._gmm.fit_predict(vectors)
            labels = self._gmm_labels

            # GMM centers are the means
            centers = self._gmm.means_
            self._n_regimes = n_clusters

            # Compute BIC for model selection
            bic = self._gmm.bic(vectors)
            aic = self._gmm.aic(vectors)
            self.logger.info(
                f"GMM: {n_clusters} regimes, BIC={bic:.1f}, AIC={aic:.1f}"
            )

        if method in ("dbscan", "both"):
            # Auto-tune eps based on nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(5, len(vectors) - 1))
            nn.fit(vectors)
            distances, _ = nn.kneighbors(vectors)
            eps = np.percentile(distances[:, -1], 90)

            self._dbscan = DBSCAN(eps=max(eps, 0.1), min_samples=3)
            self._dbscan_labels = self._dbscan.fit_predict(vectors)
            n_noise = (self._dbscan_labels == -1).sum()

            self.logger.info(
                f"DBSCAN: eps={eps:.4f}, "
                f"{len(set(self._dbscan_labels)) - (1 if -1 in self._dbscan_labels else 0)} "
                f"clusters, {n_noise} noise points"
            )

            if method == "dbscan":
                labels = self._dbscan_labels
                centers = None

        if method == "kmeans":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            centers = kmeans.cluster_centers_
            self._n_regimes = n_clusters

        if method == "both":
            # GMM labels for primary, DBSCAN for outlier detection
            outlier_mask = self._dbscan_labels == -1
            labels = self._gmm_labels.copy()
            labels[outlier_mask] = -1  # Mark outliers

        self.logger.info(
            f"Clustered {len(vectors)} vectors into "
            f"{len(set(labels)) - (1 if -1 in labels else 0)} regimes "
            f"({(labels == -1).sum()} outliers)"
        )

        return labels, centers

    # ======================================================================
    # Regime transition probability matrix
    # ======================================================================

    def compute_transition_matrix(self) -> np.ndarray:
        """Compute the regime transition probability matrix.

        P[i][j] = probability of transitioning from regime i to regime j.
        Based on the sequence of regime assignments from detect_regimes()
        and cluster_latent_vectors().

        Returns:
            Square matrix of shape (n_regimes, n_regimes).
        """
        if len(self._regime_sequence) < 2:
            self.logger.warning("Not enough regime transitions for matrix")
            return np.array([[]])

        # Get unique regime IDs
        unique_regimes = sorted(set(self._regime_sequence))
        n = len(unique_regimes)
        regime_to_idx = {r: i for i, r in enumerate(unique_regimes)}

        # Count transitions
        self._transition_counts = np.zeros((n, n))
        for t in range(len(self._regime_sequence) - 1):
            src = self._regime_sequence[t]
            dst = self._regime_sequence[t + 1]
            if src != dst:
                i = regime_to_idx[src]
                j = regime_to_idx[dst]
                self._transition_counts[i, j] += 1

        # Normalize to probabilities
        self._transition_matrix = np.zeros((n, n))
        for i in range(n):
            row_sum = self._transition_counts[i].sum()
            if row_sum > 0:
                self._transition_matrix[i] = self._transition_counts[i] / row_sum

        # Log the matrix
        self.logger.info(f"Transition matrix ({n}x{n}):")
        for i in range(n):
            probs = ", ".join(
                f"→{unique_regimes[j]}:{self._transition_matrix[i,j]:.2f}"
                for j in range(n) if self._transition_matrix[i, j] > 0
            )
            if probs:
                self.logger.info(f"  Regime {unique_regimes[i]}: {probs}")

        return self._transition_matrix

    def predict_next_regime(self, current_regime: int) -> Optional[Tuple[int, float]]:
        """Predict the most likely next regime given the current one.

        Args:
            current_regime: Current regime ID.

        Returns:
            (next_regime_id, probability) or None if no data.
        """
        if self._transition_matrix is None or self._transition_matrix.size == 0:
            self.compute_transition_matrix()

        if self._transition_matrix is None or self._transition_matrix.size == 0:
            return None

        unique_regimes = sorted(set(self._regime_sequence))
        if current_regime not in unique_regimes:
            return None

        idx = unique_regimes.index(current_regime)
        probs = self._transition_matrix[idx]

        if probs.sum() == 0:
            return None

        next_idx = int(np.argmax(probs))
        return (unique_regimes[next_idx], float(probs[next_idx]))

    def get_transition_matrix_dataframe(self) -> "pd.DataFrame":
        """Return the transition matrix as a labeled pandas DataFrame."""
        import pandas as pd
        if self._transition_matrix is None or self._transition_matrix.size == 0:
            return pd.DataFrame()
        unique_regimes = sorted(set(self._regime_sequence))
        labels = [f"R{r}" for r in unique_regimes]
        return pd.DataFrame(self._transition_matrix, index=labels, columns=labels)

    # ======================================================================
    # Alert on regime change
    # ======================================================================

    def set_alert_callback(self, callback: callable) -> None:
        """Register a callback for regime change alerts.

        Callback signature: callback(old_regime: int, new_regime: int, distance: float)
        """
        self._alert_callback = callback

    def _alert_regime_change(
        self, old_regime: int, new_regime: int, distance: float
    ) -> None:
        """Fire alert when a new regime is detected.

        Logs a warning and calls the registered callback if present.
        """
        msg = (
            f"REGIME CHANGE: {old_regime} → {new_regime} "
            f"(cosine distance: {distance:.4f})"
        )
        self.logger.warning(msg)

        if self._alert_callback is not None:
            try:
                self._alert_callback(old_regime, new_regime, distance)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    # ======================================================================
    # Regime labeling and fingerprinting
    # ======================================================================

    def label_regimes(self, labels: np.ndarray) -> Dict[int, dict]:
        """Characterize each regime with descriptive statistics.

        For each discovered regime, compute:
        - Mean latent vector (the regime "fingerprint")
        - Fraction of time spent in this regime
        - Average duration
        - Volatility proxy (latent vector std)

        Args:
            labels: Cluster labels from cluster_latent_vectors().

        Returns:
            Dict mapping regime_id to characteristic dict.
        """
        if len(self.latent_history) == 0 or len(labels) == 0:
            return {}

        vectors = np.stack(self.latent_history)
        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Skip outliers

        regime_info = {}
        total_samples = len(labels)

        for label in unique_labels:
            mask = labels == label
            regime_vectors = vectors[mask]
            n_samples = mask.sum()

            # Compute regime fingerprint (mean latent vector)
            fingerprint = regime_vectors.mean(axis=0)

            # Duration statistics
            durations = []
            current_dur = 0
            for l in labels:
                if l == label:
                    current_dur += 1
                else:
                    if current_dur > 0:
                        durations.append(current_dur)
                    current_dur = 0
            if current_dur > 0:
                durations.append(current_dur)

            avg_duration = np.mean(durations) if durations else 0

            regime_info[label] = {
                "regime_id": int(label),
                "n_samples": int(n_samples),
                "fraction": round(n_samples / max(total_samples, 1), 4),
                "avg_duration": round(float(avg_duration), 1),
                "fingerprint_norm": round(float(np.linalg.norm(fingerprint)), 4),
                "latent_std": round(float(regime_vectors.std()), 4),
                "fingerprint": fingerprint.tolist()[:10] + ["..."],  # First 10 dims
            }

        self.logger.info(f"Labeled {len(regime_info)} regimes:")
        for rid, info in regime_info.items():
            self.logger.info(
                f"  Regime {rid}: {info['n_samples']} samples, "
                f"{info['fraction']*100:.1f}% of data, "
                f"avg duration: {info['avg_duration']:.0f}"
            )

        return regime_info

    # ======================================================================
    # Real-time regime check
    # ======================================================================

    def check_current_regime(self, latent_vector: np.ndarray) -> int:
        """Assign the current latent vector to the nearest GMM regime.

        If GMM is not fitted, returns the current_regime_id from
        detect_regimes().

        Args:
            latent_vector: 256-dim latent state.

        Returns:
            Regime ID (int), or -1 for DBSCAN outliers.
        """
        if self._gmm is not None:
            vec = latent_vector.flatten().reshape(1, -1)
            gmm_label = int(self._gmm.predict(vec)[0])

            # Check if DBSCAN would flag as outlier
            if self._dbscan is not None and self._dbscan_labels is not None:
                # Compute distance to nearest core sample
                if hasattr(self._dbscan, 'core_sample_indices_') and len(self._dbscan.core_sample_indices_) > 0:
                    core_samples = np.stack(self.latent_history)[self._dbscan.core_sample_indices_]
                    min_dist = np.min(np.linalg.norm(core_samples - vec, axis=1))
                    if min_dist > self._dbscan.eps:
                        return -1

            return gmm_label

        return self.current_regime_id

    def is_new_regime(self, latent_vector: np.ndarray) -> bool:
        """Check if a latent vector represents a new (unseen) regime.

        Compares against all known regime fingerprints. If the
        minimum cosine distance exceeds threshold, it's a new regime.

        Args:
            latent_vector: 256-dim latent state.

        Returns:
            True if this is a previously unseen regime.
        """
        if len(self.latent_history) < 20:
            return False

        vec = latent_vector.flatten()
        history = np.stack(self.latent_history[-100:])

        # Compare against the rolling mean
        rolling_mean = history.mean(axis=0)
        distance = cosine(rolling_mean, vec)

        return distance > self.threshold

    # ======================================================================
    # Helper: encode price window to latent
    # ======================================================================

    def _encode_price_window(self, window: np.ndarray) -> Optional[np.ndarray]:
        """Encode a price window to a latent vector using the world model."""
        try:
            window = window[-500:].astype(np.float32)
            if len(window) < 500:
                window = np.pad(window, (500 - len(window), 0), mode="edge")

            mean = window.mean()
            std = window.std()
            if std < 1e-8:
                std = 1.0
            normalized = (window - mean) / std

            x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            x = x.to(self.device)

            with torch.no_grad():
                mu, logvar, z = self.world_model.encode(x)

            return z.cpu().numpy().flatten()
        except Exception as e:
            self.logger.warning(f"Failed to encode window: {e}")
            return None

    # ======================================================================
    # Accessors
    # ======================================================================

    def get_current_regime(self) -> int:
        """Return the current regime ID."""
        return self.current_regime_id

    def get_regime_at_index(self, idx: int) -> int:
        """Return the regime ID at a given price index."""
        regime_id = 0
        for b in self.regime_boundaries:
            if idx < b["index"]:
                break
            regime_id = b["regime_id"]
        return regime_id

    def get_regime_statistics(self) -> dict:
        """Compute statistics about detected regimes."""
        if not self.regime_boundaries:
            return {
                "num_regimes": 1,
                "avg_regime_length": 0,
                "max_regime_length": 0,
                "min_regime_length": 0,
                "boundaries": [],
                "total_boundaries": 0,
                "transition_matrix": None,
            }

        lengths = []
        prev_idx = 0
        stats_boundaries = []

        for b in self.regime_boundaries:
            length = b["index"] - prev_idx
            if length >= self.MIN_REGIME_DURATION:
                lengths.append(length)
                stats_boundaries.append({
                    "regime_id": b["regime_id"],
                    "start_index": prev_idx,
                    "length": length,
                    "distance": b["distance"],
                })
            prev_idx = b["index"]

        transition_matrix = None
        if self._transition_matrix is not None and self._transition_matrix.size > 0:
            transition_matrix = self._transition_matrix.tolist()

        return {
            "num_regimes": len(lengths) + 1,
            "avg_regime_length": float(np.mean(lengths)) if lengths else 0,
            "max_regime_length": int(np.max(lengths)) if lengths else 0,
            "min_regime_length": int(np.min(lengths)) if lengths else 0,
            "boundaries": stats_boundaries,
            "total_boundaries": len(self.regime_boundaries),
            "transition_matrix": transition_matrix,
        }

    def get_regime_fingerprints(self) -> Optional[np.ndarray]:
        """Return the GMM regime fingerprints (mean vectors)."""
        if self._gmm is not None:
            return self._gmm.means_
        return None
