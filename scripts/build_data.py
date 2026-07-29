#!/usr/bin/env python3
"""Build DriftWatch's deterministic browser artifacts from the UCI dataset."""

from __future__ import annotations

import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 202707
BATCH_SIZE = 400
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/online_shoppers_intention.csv"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FIT_MONTHS = MONTHS[:3]
REFERENCE_MONTH = "June"
STREAM_MONTHS = MONTHS[4:]

NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
CATEGORICAL = [
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURES = NUMERIC + CATEGORICAL
LABELS = {
    "Administrative": "Administrative pages",
    "Administrative_Duration": "Administrative duration",
    "Informational": "Informational pages",
    "Informational_Duration": "Informational duration",
    "ProductRelated": "Product pages",
    "ProductRelated_Duration": "Product duration",
    "BounceRates": "Bounce rate",
    "ExitRates": "Exit rate",
    "PageValues": "Page value",
    "SpecialDay": "Special day proximity",
    "OperatingSystems": "Operating system",
    "Browser": "Browser",
    "Region": "Region",
    "TrafficType": "Traffic type",
    "VisitorType": "Visitor type",
    "Weekend": "Weekend",
}


def download_raw() -> None:
    if RAW_PATH.exists():
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading UCI dataset to {RAW_PATH.relative_to(ROOT)}")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        RAW_PATH.write_bytes(zipped.read("online_shoppers_intention.csv"))


def histogram_edges(values: np.ndarray) -> list[float]:
    """Reference-decile edges; under/overflow values are folded into end bins."""
    edges = np.unique(np.quantile(values.astype(float), np.linspace(0, 1, 11)))
    if len(edges) < 2:
        value = float(edges[0])
        edges = np.array([value - 0.5, value + 0.5])
    return [round(float(value), 8) for value in edges]


def histogram_counts(values: np.ndarray, edges: list[float]) -> list[int]:
    bins = np.searchsorted(np.asarray(edges[1:-1]), values.astype(float), side="right")
    return np.bincount(bins, minlength=len(edges) - 1).astype(int).tolist()


def feature_summary(frame: pd.DataFrame, edges: dict[str, list[float]]) -> dict:
    summary = {}
    for name in NUMERIC:
        values = frame[name].to_numpy(dtype=float)
        summary[name] = {
            "type": "numeric",
            "edges": edges[name],
            "counts": histogram_counts(values, edges[name]),
            "values": np.round(values, 6).tolist(),
        }
    for name in CATEGORICAL:
        counts = frame[name].astype(str).value_counts(sort=False)
        summary[name] = {
            "type": "categorical",
            "counts": {key: int(value) for key, value in counts.items()},
        }
    return summary


def performance(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(labels, probabilities >= 0.5)), 6),
        "auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "brier": round(float(brier_score_loss(labels, probabilities)), 6),
        "positiveRate": round(float(np.mean(labels)), 6),
    }


def json_write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def batch_frames(stream: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    batches = []
    for month in STREAM_MONTHS:
        month_rows = stream[stream["Month"] == month]
        for chunk, start in enumerate(range(0, len(month_rows), BATCH_SIZE), 1):
            frame = month_rows.iloc[start : start + BATCH_SIZE].copy()
            batches.append((f"{month.lower()}-{chunk:02d}", f"{month} {chunk:02d}", frame))
    return batches


def serialize_batch(
    batch_id: str,
    label: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    edges: dict[str, list[float]],
) -> dict:
    return {
        "id": batch_id,
        "label": label,
        "month": str(frame["Month"].iloc[0]),
        "size": len(frame),
        "predictions": np.round(probabilities, 6).tolist(),
        "labels": labels.astype(int).tolist(),
        "features": feature_summary(frame, edges),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    download_raw()

    data = pd.read_csv(RAW_PATH)
    assert len(data) == 12_330 and not data.isna().any().any()
    data["_source_order"] = np.arange(len(data))
    data["_month_order"] = data["Month"].map({month: i for i, month in enumerate(MONTHS)})
    data = data.sort_values(["_month_order", "_source_order"], kind="stable")
    data["Revenue"] = data["Revenue"].astype(int)

    fit = data[data["Month"].isin(FIT_MONTHS)].copy()
    reference = data[data["Month"] == REFERENCE_MONTH].copy()
    stream = data[data["Month"].isin(STREAM_MONTHS)].copy()
    assert (len(fit), len(reference), len(stream)) == (5_455, 288, 6_587)

    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = make_pipeline(
        preprocess,
        LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
            solver="liblinear",
        ),
    )
    model.fit(fit[FEATURES], fit["Revenue"])

    reference_probabilities = model.predict_proba(reference[FEATURES])[:, 1]
    reference_labels = reference["Revenue"].to_numpy(dtype=int)
    reference_metrics = performance(reference_labels, reference_probabilities)
    edges = {
        name: histogram_edges(reference[name].to_numpy(dtype=float)) for name in NUMERIC
    }
    feature_meta = [
        {"name": name, "label": LABELS[name], "type": "numeric" if name in NUMERIC else "categorical"}
        for name in FEATURES
    ]
    common_meta = {
        "dataset": "UCI Online Shoppers Purchasing Intention",
        "datasetDoi": "10.24432/C5F88Q",
        "license": "CC BY 4.0",
        "model": "Balanced logistic regression",
        "seed": SEED,
        "fitWindow": "Feb, Mar, May",
        "fitRows": len(fit),
        "referenceWindow": REFERENCE_MONTH,
        "referenceRows": len(reference),
        "streamWindow": "Jul-Dec",
        "streamRows": len(stream),
        "batchSize": BATCH_SIZE,
        "features": feature_meta,
    }
    reference_payload = {
        "meta": common_meta,
        "performance": reference_metrics,
        "predictions": np.round(reference_probabilities, 6).tolist(),
        "labels": reference_labels.tolist(),
        "features": feature_summary(reference, edges),
    }

    frames = batch_frames(stream)
    real_batches = []
    for batch_id, label, frame in frames:
        probabilities = model.predict_proba(frame[FEATURES])[:, 1]
        labels = frame["Revenue"].to_numpy(dtype=int)
        real_batches.append(
            serialize_batch(batch_id, label, frame, probabilities, labels, edges)
        )

    onset = len(frames) // 2
    rng = np.random.default_rng(SEED)
    injected_batches = []
    flipped = 0
    for index, (batch_id, label, source_frame) in enumerate(frames):
        frame = source_frame.copy()
        labels = frame["Revenue"].to_numpy(dtype=int).copy()
        if index >= onset:
            frame["ProductRelated_Duration"] = (
                frame["ProductRelated_Duration"] * 1.6
                + rng.normal(240, 25, size=len(frame))
            ).clip(lower=0)
            frame["ExitRates"] = (frame["ExitRates"] + 0.035).clip(upper=1)
            candidates = frame["VisitorType"].eq("Returning_Visitor").to_numpy()
            flip = candidates & (rng.random(len(frame)) < 0.65)
            labels[flip] = 1 - labels[flip]
            flipped += int(flip.sum())
        probabilities = model.predict_proba(frame[FEATURES])[:, 1]
        injected_batches.append(
            serialize_batch(batch_id, label, frame, probabilities, labels, edges)
        )

    stream_meta = {
        "real": {
            "kind": "real",
            "label": "Real months",
            "description": "Observed UCI sessions in chronological month order.",
        },
        "injected": {
            "kind": "injected",
            "label": "Synthetic drift, injected for demonstration",
            "description": (
                "From the midpoint onward: ProductRelated_Duration is multiplied by 1.6 "
                "plus seeded noise, ExitRates increases by 0.035, and 65% of returning-"
                "visitor labels are flipped with seed 202707."
            ),
            "onsetBatch": frames[onset][0],
            "flippedLabels": flipped,
        },
    }
    json_write(ROOT / "data/reference.json", reference_payload)
    json_write(
        ROOT / "data/real.json",
        {**stream_meta["real"], "batches": real_batches},
    )
    json_write(
        ROOT / "data/injected.json",
        {**stream_meta["injected"], "batches": injected_batches},
    )

    data_files = list((ROOT / "data").rglob("*"))
    total_bytes = sum(path.stat().st_size for path in data_files if path.is_file())
    assert len(real_batches) == len(injected_batches) == 21
    assert all(
        batch["size"] == len(batch["labels"]) == len(batch["predictions"])
        for batch in real_batches + injected_batches
    )
    assert all(
        sum(
            feature["counts"].values()
            if isinstance(feature["counts"], dict)
            else feature["counts"]
        )
        == batch["size"]
        for batch in real_batches + injected_batches
        for feature in batch["features"].values()
    )
    assert flipped > 0
    assert total_bytes < 5 * 1024 * 1024, f"data artifacts too large: {total_bytes:,} bytes"

    print(
        "SUCCESS CHECK:",
        f"fit={len(fit):,}",
        f"reference={len(reference):,}",
        f"stream={len(stream):,}",
        f"batches={len(real_batches)}",
        f"injected_labels_flipped={flipped:,}",
        f"data_bytes={total_bytes:,}",
    )
    print(
        "REFERENCE METRICS:",
        " ".join(f"{name}={value:.4f}" for name, value in reference_metrics.items()),
    )


if __name__ == "__main__":
    main()
