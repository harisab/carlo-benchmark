from __future__ import annotations

from typing import Any

import numpy as np

from odmr.algorithms.common import merged_settings, y_dip_to_counts_like

try:
    from sklearn.cluster import KMeans

    SKLEARN_OK = True
except Exception:
    KMeans = None
    SKLEARN_OK = False


def _contains(a: np.ndarray, b: np.ndarray, threshold_percentage: float) -> bool:
    if len(a) == 0:
        return False

    count = np.isin(a, b).sum()
    percent = (count / len(a)) * 100.0
    return bool(percent >= threshold_percentage)


def _fit_kmeans(values: np.ndarray, *, n_clusters: int, max_iter: int, random_state: int | None):
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn is required for Paper CA. Install scikit-learn.")

    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=max_iter,
        algorithm="lloyd",
        random_state=random_state,
    ).fit(values.reshape(-1, 1))


def _safe_width(mw_subset: np.ndarray) -> float:
    if len(mw_subset) == 0:
        return 0.0
    return float(np.abs(np.max(mw_subset) - np.min(mw_subset)))


def _paper_ca_core(
    mw: np.ndarray,
    y_dip: np.ndarray,
    *,
    k_y: int,
    max_iter: int,
    clean_width_bug: bool,
    random_state: int | None,
) -> tuple[float, float]:
    """
    Single-trace adaptation of Dylan Stone's clustering algorithm.

    Algorithm idea:
    1. KMeans cluster vertical ODMR values into k_y levels.
    2. Take the lowest cluster, then progressively include more low clusters.
    3. For each selected low-value subset, KMeans cluster the MW positions into 2 groups.
    4. Use the two horizontal centroids as resonance estimates.

    clean_width_bug=False preserves the reference behavior where width2 is
    computed from cluster1 again.
    """
    mw = np.asarray(mw, dtype=float)
    y_for_clustering = y_dip_to_counts_like(y_dip)

    if len(mw) != len(y_for_clustering):
        raise ValueError("mw and y_dip must have the same length.")

    if k_y < 2:
        raise ValueError("paper_ca_k_y must be >= 2.")

    if len(mw) < k_y:
        raise ValueError("Trace has fewer points than paper_ca_k_y.")

    row_model = _fit_kmeans(
        y_for_clustering,
        n_clusters=int(k_y),
        max_iter=int(max_iter),
        random_state=random_state,
    )

    labels = row_model.labels_
    cluster_centers = row_model.cluster_centers_[:, 0]

    indice_options: list[np.ndarray] = []
    model_x_options: list[Any] = []
    peaks: list[np.ndarray] = []
    lowest = True

    # Initial lowest vertical cluster.
    cluster_indices = np.argsort(cluster_centers)[:1]
    lowest_indices = np.concatenate(
        [np.where(labels == cluster_idx)[0] for cluster_idx in cluster_indices]
    )
    indice_options.append(lowest_indices)

    try:
        model_x = _fit_kmeans(
            mw[indice_options[0]],
            n_clusters=2,
            max_iter=300,
            random_state=random_state,
        )
        model_x_options.append(model_x)
    except ValueError:
        model_x_options.append([])

    for j in range(k_y - 1):
        # Add one more low vertical cluster.
        cluster_indices = np.argsort(cluster_centers)[: (j + 2)]
        selected_indices = np.concatenate(
            [np.where(labels == cluster_idx)[0] for cluster_idx in cluster_indices]
        )
        indice_options.append(selected_indices)

        try:
            model_x = _fit_kmeans(
                mw[indice_options[j + 1]],
                n_clusters=2,
                max_iter=300,
                random_state=random_state,
            )
            model_x_options.append(model_x)
        except ValueError:
            model_x_options.append([])

        if isinstance(model_x_options[j], list):
            continue

        current_model = model_x_options[j]
        next_model = model_x_options[j + 1]

        if isinstance(next_model, list):
            continue

        # Current model widths.
        cluster1 = np.where(current_model.labels_ == 0)[0]
        cluster2 = np.where(current_model.labels_ == 1)[0]

        width1 = _safe_width(mw[cluster1])
        if clean_width_bug:
            width2 = _safe_width(mw[cluster2])
        else:
            # Verbatim Dylan/reference behavior: width2 also uses cluster1.
            width2 = _safe_width(mw[cluster1])

        width_current = (width1 + width2) / 2.0

        centroids = current_model.cluster_centers_[:, 0]
        cluster_distance_current = float(np.abs(centroids[0] - centroids[1]))

        # Next model widths.
        cluster1 = np.where(next_model.labels_ == 0)[0]
        cluster2 = np.where(next_model.labels_ == 1)[0]

        width1 = _safe_width(mw[cluster1])
        if clean_width_bug:
            width2 = _safe_width(mw[cluster2])
        else:
            width2 = _safe_width(mw[cluster1])

        width_next = (width1 + width2) / 2.0

        centroids = next_model.cluster_centers_[:, 0]
        cluster_distance_next = float(np.abs(centroids[0] - centroids[1]))

        if (
            cluster_distance_current <= 0.0
            or width_current <= 0.0
            or width_next <= 0.0
        ):
            comparison = np.inf
        else:
            comparison = (
                (cluster_distance_next / cluster_distance_current)
                / (width_next / width_current)
            )

        if comparison <= 1.0:
            if lowest is False:
                prev_model = model_x_options[j - 1]

                prev_clust1 = np.where(prev_model.labels_ == 0)[0]
                prev_clust2 = np.where(prev_model.labels_ == 1)[0]

                cluster1 = np.where(current_model.labels_ == 0)[0]
                cluster2 = np.where(current_model.labels_ == 1)[0]

                if _contains(prev_clust1, cluster1, 80):
                    cluster1 = np.concatenate((prev_clust1, prev_clust2))
                    recluster_indices = np.concatenate((cluster1, cluster2))

                    model_x_new = _fit_kmeans(
                        mw[indice_options[j]][recluster_indices],
                        n_clusters=2,
                        max_iter=300,
                        random_state=random_state,
                    )
                    peaks.append(model_x_new.cluster_centers_)
                    break

                if _contains(prev_clust1, cluster2, 80):
                    cluster2 = np.concatenate((prev_clust1, prev_clust2))
                    recluster_indices = np.concatenate((cluster1, cluster2))

                    model_x_new = _fit_kmeans(
                        mw[indice_options[j]][recluster_indices],
                        n_clusters=2,
                        max_iter=300,
                        random_state=random_state,
                    )
                    peaks.append(model_x_new.cluster_centers_)
                    break

                peaks.append(prev_model.cluster_centers_)
                break

            peaks.append(current_model.cluster_centers_)
            break

        lowest = False

        # If peaks are never selected and we are on the last iteration,
        # use the first valid horizontal model.
        if j == (k_y - 2):
            for model in model_x_options:
                if not isinstance(model, list):
                    peaks.append(model.cluster_centers_)
                    break

            if not peaks:
                peaks.append(np.array([[np.nan], [np.nan]], dtype=float))

    if not peaks:
        peaks.append(np.array([[np.nan], [np.nan]], dtype=float))

    peaks_arr = np.asarray(peaks[0], dtype=float).reshape(-1)
    f1_hat = float(np.nanmin(peaks_arr))
    f2_hat = float(np.nanmax(peaks_arr))

    return f1_hat, f2_hat


def run_paper_ca_verbatim(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    settings: dict[str, Any] | None = None,
) -> dict:
    cfg = merged_settings(settings)

    f1_hat, f2_hat = _paper_ca_core(
        np.asarray(x, dtype=float),
        np.asarray(y_dip, dtype=float),
        k_y=int(cfg.get("paper_ca_k_y", 4)),
        max_iter=int(cfg.get("paper_ca_max_iter", 300)),
        clean_width_bug=False,
        random_state=None,
    )

    return {
        "name": "PaperCA_Verbatim",
        "benchmark_variant": str(cfg.get("benchmark_variant", "paper_ca_verbatim")),
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma": float("nan"),
        "score": float("nan"),
        "used_settings": cfg,
        "best_fit": None,
    }


def run_paper_ca_clean(
    x: np.ndarray,
    y_dip: np.ndarray,
    *,
    settings: dict[str, Any] | None = None,
) -> dict:
    cfg = merged_settings(settings)

    random_state_raw = cfg.get("paper_ca_random_state", 0)
    random_state = None if random_state_raw is None else int(random_state_raw)

    f1_hat, f2_hat = _paper_ca_core(
        np.asarray(x, dtype=float),
        np.asarray(y_dip, dtype=float),
        k_y=int(cfg.get("paper_ca_k_y", 4)),
        max_iter=int(cfg.get("paper_ca_max_iter", 300)),
        clean_width_bug=True,
        random_state=random_state,
    )

    return {
        "name": "PaperCA_Clean",
        "benchmark_variant": str(cfg.get("benchmark_variant", "paper_ca_clean")),
        "f1_hat": f1_hat,
        "f2_hat": f2_hat,
        "gamma": float("nan"),
        "score": float("nan"),
        "used_settings": cfg,
        "best_fit": None,
    }