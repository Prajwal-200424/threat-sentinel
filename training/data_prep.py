"""
Threat Sentinel - leakage-safe N-BaIoT data preparation.

Key guarantees:
1. X, y, family and stream/device metadata always stay aligned.
2. The held-out attack family is selected BEFORE any balancing.
3. Balancing is applied only to the training rows used by RF/AE.
4. Benign data is split chronologically per source stream so the same rows
   are never present in both train and test.
5. Sequence windows are created independently inside each source stream;
   windows never cross CSV/device boundaries.
6. Training/calibration sequences are split chronologically, preventing
   overlapping-window leakage between fit and calibration.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


VALID_FAMILIES = {"BENIGN", "GAFGYT", "MIRAI"}


def _subsample_preserve_order(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df.reset_index(drop=True)
    if frac <= 0.0:
        raise ValueError("subsample_frac must be > 0")

    n = max(1, int(round(len(df) * frac)))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))
    return df.iloc[selected].reset_index(drop=True)


def load_nbaiot_data(data_dir: str, subsample_frac: float = 0.2, random_state: int = 42) -> dict:
    """Load CSVs while preserving row order and source-stream metadata."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    records = []
    feature_columns = None

    print(f"Found {len(csv_files)} CSV files")

    for file_index, csv_file in enumerate(csv_files):
        filename = csv_file.name.lower()

        if filename in {"data_summary.csv", "device_info.csv", "features.csv"}:
            continue

        if "benign" in filename:
            family = "BENIGN"
            label = 0
        elif "gafgyt" in filename:
            family = "GAFGYT"
            label = 1
        elif "mirai" in filename:
            family = "MIRAI"
            label = 1
        else:
            continue

        df = pd.read_csv(csv_file)
        if df.empty:
            continue

        df = _subsample_preserve_order(df, subsample_frac, random_state + file_index)

        if feature_columns is None:
            feature_columns = list(df.columns)
        else:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                raise ValueError(f"{csv_file.name} is missing feature columns: {missing[:5]}")

        # Keep only the common feature columns and coerce safely to numeric.
        features = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_file = features.to_numpy(dtype=np.float32)

        # Each CSV is treated as an independent temporal stream.
        stream_id = csv_file.stem
        family_arr = np.full(len(df), family, dtype=object)
        label_arr = np.full(len(df), label, dtype=np.int8)
        stream_arr = np.full(len(df), stream_id, dtype=object)

        records.append(
            {
                "X": X_file,
                "y": label_arr,
                "family": family_arr,
                "stream": stream_arr,
                "source_file": csv_file.name,
            }
        )

        print(
            f"  Loaded {csv_file.name}: {len(df)} records "
            f"(family={family}, label={label})"
        )

    if not records:
        raise ValueError(f"No supported N-BaIoT CSV files found in {data_dir}")

    X = np.vstack([r["X"] for r in records])
    y = np.concatenate([r["y"] for r in records]).astype(int)
    family = np.concatenate([r["family"] for r in records]).astype(object)
    stream = np.concatenate([r["stream"] for r in records]).astype(object)

    print(f"\nTotal records: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Benign: {(y == 0).sum()}")
    print(f"Attack: {(y == 1).sum()}")
    print(f"Gafgyt: {(family == 'GAFGYT').sum()}")
    print(f"Mirai: {(family == 'MIRAI').sum()}")

    return {
        "X": X,
        "y": y,
        "family": family,
        "stream": stream,
        "feature_columns": feature_columns,
    }


def _split_benign_by_stream(
    X: np.ndarray,
    y: np.ndarray,
    family: np.ndarray,
    stream: np.ndarray,
    test_fraction: float = 0.20,
):
    """Chronologically split each benign source stream into train/test."""
    train_parts = []
    test_parts = []

    benign_streams = np.unique(stream[family == "BENIGN"])

    for stream_id in benign_streams:
        idx = np.where((family == "BENIGN") & (stream == stream_id))[0]
        if len(idx) < 2:
            train_parts.append(idx)
            continue

        split = int(np.floor(len(idx) * (1.0 - test_fraction)))
        split = min(max(split, 1), len(idx) - 1)
        train_parts.append(idx[:split])
        test_parts.append(idx[split:])

    return np.concatenate(train_parts) if train_parts else np.array([], dtype=int), \
        np.concatenate(test_parts) if test_parts else np.array([], dtype=int)


def _split_train_calibration_by_stream(
    train_indices: np.ndarray,
    stream: np.ndarray,
    calibration_fraction: float = 0.20,
):
    """Chronologically split training rows into fit/calibration per stream."""
    fit_parts = []
    cal_parts = []

    for stream_id in np.unique(stream[train_indices]):
        idx = train_indices[stream[train_indices] == stream_id]
        if len(idx) < 2:
            fit_parts.append(idx)
            continue

        split = int(np.floor(len(idx) * (1.0 - calibration_fraction)))
        split = min(max(split, 1), len(idx) - 1)
        fit_parts.append(idx[:split])
        cal_parts.append(idx[split:])

    fit = np.concatenate(fit_parts) if fit_parts else np.array([], dtype=int)
    cal = np.concatenate(cal_parts) if cal_parts else np.array([], dtype=int)
    return fit, cal


def _pack(X, y, family, stream, indices):
    return {
        "X": X[indices],
        "y": y[indices],
        "family": family[indices],
        "stream": stream[indices],
        "indices": np.asarray(indices, dtype=int),
    }


def balance_training_data(X, y, family, stream, random_state: int = 42):
    """Balance ONLY the flat training fit set, preserving metadata alignment."""
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2:
        raise ValueError(f"Expected both classes in training data, got {unique}")

    min_count = int(counts.min())
    rng = np.random.default_rng(random_state)
    selected_parts = []

    for class_label in sorted(unique):
        indices = np.where(y == class_label)[0]
        selected = rng.choice(indices, size=min_count, replace=False)
        selected_parts.append(selected)

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)

    X_bal = X[selected]
    y_bal = y[selected]
    family_bal = family[selected]
    stream_bal = stream[selected]

    if not np.array_equal(y_bal == 0, family_bal == "BENIGN"):
        raise RuntimeError("Label/family alignment was lost during balancing")

    print("\nBalanced TRAIN FIT set:")
    print(f"Benign: {(y_bal == 0).sum()}")
    print(f"Attack: {(y_bal == 1).sum()}")
    print(f"Families: {dict(zip(*np.unique(family_bal, return_counts=True)))}")

    return X_bal, y_bal, family_bal, stream_bal


def prepare_data(
    data_dir: str,
    subsample_frac: float = 0.2,
    test_size: float = 0.20,
    calibration_size: float = 0.20,
    random_state: int = 42,
    holdout_family: Optional[str] = "MIRAI",
) -> dict:
    """Create a leakage-safe family-held-out dataset."""
    if holdout_family not in VALID_FAMILIES - {"BENIGN"}:
        raise ValueError(f"holdout_family must be GAFGYT or MIRAI, got {holdout_family}")

    raw = load_nbaiot_data(data_dir, subsample_frac, random_state)
    X = raw["X"]
    y = raw["y"]
    family = raw["family"]
    stream = raw["stream"]

    print("\nCreating family-aware split BEFORE balancing...")

    # Attack split is family-based: all held-out-family attacks are test only.
    attack_train_idx = np.where(
        (y == 1) & (family != holdout_family)
    )[0]
    attack_test_idx = np.where(
        (y == 1) & (family == holdout_family)
    )[0]

    # Benign data is split chronologically per source stream.
    benign_train_idx, benign_test_idx = _split_benign_by_stream(
        X, y, family, stream, test_fraction=test_size
    )

    train_pool_idx = np.concatenate([benign_train_idx, attack_train_idx])
    test_idx = np.concatenate([benign_test_idx, attack_test_idx])

    # Stable chronological ordering by original row index.
    train_pool_idx.sort()
    test_idx.sort()

    if len(attack_train_idx) == 0 or len(attack_test_idx) == 0:
        raise RuntimeError("Family-aware split produced an empty attack partition")
    if len(benign_train_idx) == 0 or len(benign_test_idx) == 0:
        raise RuntimeError("Benign split produced an empty train/test partition")

    # Chronological fit/calibration split within each training stream.
    fit_idx, cal_idx = _split_train_calibration_by_stream(
        train_pool_idx, stream, calibration_fraction=calibration_size
    )
    fit_idx.sort()
    cal_idx.sort()

    # Fit scaler on FIT ONLY. Calibration and test never influence scaling.
    scaler = StandardScaler()
    scaler.fit(X[fit_idx])

    X_fit_raw = X[fit_idx]
    X_cal_raw = X[cal_idx]
    X_test_raw = X[test_idx]

    X_fit = scaler.transform(X_fit_raw).astype(np.float32)
    X_cal = scaler.transform(X_cal_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    y_fit = y[fit_idx].astype(int)
    y_cal = y[cal_idx].astype(int)
    y_test = y[test_idx].astype(int)

    family_fit = family[fit_idx]
    family_cal = family[cal_idx]
    family_test = family[test_idx]
    stream_fit = stream[fit_idx]
    stream_cal = stream[cal_idx]
    stream_test = stream[test_idx]

    # Balance only the flat FIT data used by RF/AE.
    X_train, y_train, family_train, stream_train = balance_training_data(
        X_fit, y_fit, family_fit, stream_fit, random_state=random_state
    )

    print("\nZERO-DAY SPLIT")
    print(f"Hold-out family: {holdout_family}")
    print(f"TRAIN FIT rows: {len(X_train)}")
    print(f"CALIBRATION rows: {len(X_cal)}")
    print(f"TEST rows: {len(X_test)}")
    print(f"Train FIT families: {np.unique(family_train)}")
    print(f"Calibration families: {np.unique(family_cal)}")
    print(f"Test families: {np.unique(family_test)}")

    if holdout_family in set(family_train) or holdout_family in set(family_cal):
        raise RuntimeError("HOLDOUT FAMILY LEAKED INTO TRAIN/CALIBRATION")
    if np.any(family_test != "BENIGN") and not np.all(
        np.isin(family_test, ["BENIGN", holdout_family])
    ):
        raise RuntimeError("Unexpected family in test set")

    return {
        # Flat fit set: balanced, for RF and AE.
        "X_train": X_train,
        "y_train": y_train,
        "family_train": family_train,
        "stream_train": stream_train,

        # Flat calibration: unbalanced, chronological, never used for fitting.
        "X_cal": X_cal,
        "y_cal": y_cal,
        "family_cal": family_cal,
        "stream_cal": stream_cal,

        # Flat test: benign + held-out attack family.
        "X_test": X_test,
        "y_test": y_test,
        "family_test": family_test,
        "stream_test": stream_test,

        # Raw/scaled stream arrays used to construct non-crossing sequences.
        "sequence_fit": {
            "X": X_fit,
            "y": y_fit,
            "family": family_fit,
            "stream": stream_fit,
        },
        "sequence_cal": {
            "X": X_cal,
            "y": y_cal,
            "family": family_cal,
            "stream": stream_cal,
        },
        "sequence_test": {
            "X": X_test,
            "y": y_test,
            "family": family_test,
            "stream": stream_test,
        },
        "scaler": scaler,
        "holdout_family": holdout_family,
        "feature_columns": raw["feature_columns"],
    }


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    family: np.ndarray,
    stream: np.ndarray,
    window_size: int = 10,
):
    """Create windows independently within each stream."""
    X_seq_parts = []
    y_seq_parts = []
    family_seq_parts = []
    stream_seq_parts = []
    target_indices = []

    if len(X) != len(y) or len(X) != len(family) or len(X) != len(stream):
        raise ValueError("X/y/family/stream lengths do not match")

    for stream_id in np.unique(stream):
        idx = np.where(stream == stream_id)[0]
        if len(idx) < window_size:
            continue

        # idx is already chronological because input preserves source order.
        for local_i in range(len(idx) - window_size + 1):
            window_idx = idx[local_i : local_i + window_size]
            target_idx = window_idx[-1]

            X_seq_parts.append(X[window_idx])
            y_seq_parts.append(int(y[target_idx]))
            family_seq_parts.append(family[target_idx])
            stream_seq_parts.append(stream_id)
            target_indices.append(int(target_idx))

    if not X_seq_parts:
        raise ValueError(f"No sequences could be created with window_size={window_size}")

    return {
        "X": np.asarray(X_seq_parts, dtype=np.float32),
        "y": np.asarray(y_seq_parts, dtype=int),
        "family": np.asarray(family_seq_parts, dtype=object),
        "stream": np.asarray(stream_seq_parts, dtype=object),
        "target_indices": np.asarray(target_indices, dtype=int),
    }


def prepare_sequences(data: dict, window_size: int = 10) -> dict:
    """Prepare leakage-safe sequence sets with exact target metadata."""
    fit = data["sequence_fit"]
    cal = data["sequence_cal"]
    test = data["sequence_test"]

    fit_seq = create_sequences(
        fit["X"], fit["y"], fit["family"], fit["stream"], window_size
    )
    cal_seq = create_sequences(
        cal["X"], cal["y"], cal["family"], cal["stream"], window_size
    )
    test_seq = create_sequences(
        test["X"], test["y"], test["family"], test["stream"], window_size
    )

    # Sequence labels must match the last element of each sequence.
    for name, seq in [("fit", fit_seq), ("calibration", cal_seq), ("test", test_seq)]:
        if not np.all(
            seq["y"]
            == np.array(
                [fit["y"] if name == "fit" else cal["y"] if name == "calibration" else test["y"]][0][seq["target_indices"]]
            )
        ):
            raise RuntimeError(f"{name} sequence target alignment failed")

    print("\nSEQUENCE SETS")
    print(f"Train sequences: {len(fit_seq['X'])}")
    print(f"Calibration sequences: {len(cal_seq['X'])}")
    print(f"Test sequences: {len(test_seq['X'])}")
    print(f"Sequence shape: {fit_seq['X'].shape[1:]}")

    return {
        "X_train": fit_seq["X"],
        "y_train": fit_seq["y"],
        "family_train": fit_seq["family"],
        "stream_train": fit_seq["stream"],
        "target_indices_train": fit_seq["target_indices"],

        "X_cal": cal_seq["X"],
        "y_cal": cal_seq["y"],
        "family_cal": cal_seq["family"],
        "stream_cal": cal_seq["stream"],
        "target_indices_cal": cal_seq["target_indices"],

        "X_test": test_seq["X"],
        "y_test": test_seq["y"],
        "family_test": test_seq["family"],
        "stream_test": test_seq["stream"],
        "target_indices_test": test_seq["target_indices"],

        "scaler": data["scaler"],
        "holdout_family": data["holdout_family"],
    }


if __name__ == "__main__":
    archive_path = r"C:\Users\Tejar\Downloads\archive"
    data = prepare_data(archive_path, subsample_frac=0.2, holdout_family="MIRAI")
    seq = prepare_sequences(data, window_size=10)
    print("\nData preparation complete.")
    print("Flat train:", data["X_train"].shape)
    print("Flat calibration:", data["X_cal"].shape)
    print("Flat test:", data["X_test"].shape)
    print("Sequence train:", seq["X_train"].shape)
    print("Sequence calibration:", seq["X_cal"].shape)
    print("Sequence test:", seq["X_test"].shape)
