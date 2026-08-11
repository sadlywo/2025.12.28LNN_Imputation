"""Create a publication figure of representative trajectories from three corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_v2.data import get_dataset_adapter


# Editable text in SVG/PDF is mandatory for manuscript production.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 6.5
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.major.size"] = 2.5
plt.rcParams["ytick.major.size"] = 2.5
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 116.0
MAX_PLOTTED_POINTS = 2600
CMAP = mpl.colormaps["viridis"]
START_COLOR = "#2E9E44"
END_COLOR = "#D9473F"


SELECTION = (
    {
        "dataset": "OxIOD",
        "domain": "smartphone motion",
        "adapter": "oxiod",
        "root": "Oxford Dataset",
        "recording_id": "handbag-1/imu1",
        "label": "Handbag 1",
    },
    {
        "dataset": "OxIOD",
        "domain": "smartphone motion",
        "adapter": "oxiod",
        "root": "Oxford Dataset",
        "recording_id": "running/imu1",
        "label": "Running",
    },
    {
        "dataset": "EuRoC MAV",
        "domain": "micro aerial vehicle",
        "adapter": "euroc_mav",
        "root": "datasets/processed/euroc_mav",
        "recording_id": "euroc_mav/V1_01_easy",
        "label": "V1-01 Easy",
    },
    {
        "dataset": "EuRoC MAV",
        "domain": "micro aerial vehicle",
        "adapter": "euroc_mav",
        "root": "datasets/processed/euroc_mav",
        "recording_id": "euroc_mav/V2_02_medium",
        "label": "V2-02 Medium",
    },
    {
        "dataset": "IDOL",
        "domain": "indoor pedestrian",
        "adapter": "idol",
        "root": "datasets/raw/idol/archives",
        "recording_id": "idol/building1/train/17",
        "label": "Building 1 · train 17",
    },
    {
        "dataset": "IDOL",
        "domain": "indoor pedestrian",
        "adapter": "idol",
        "root": "datasets/raw/idol/archives",
        "recording_id": "idol/building3/known/7",
        "label": "Building 3 · known 7",
    },
)


def _load_selected() -> tuple[list[dict[str, object]], dict[str, int]]:
    discovered: dict[tuple[str, str], dict[str, str]] = {}
    corpus_counts: dict[str, int] = {}
    loaded: list[dict[str, object]] = []
    for item in SELECTION:
        adapter_name = str(item["adapter"])
        root = (ROOT / str(item["root"])).resolve()
        key = (adapter_name, str(root))
        if key not in discovered:
            adapter = get_dataset_adapter(adapter_name)
            pairs = [dict(pair) for pair in adapter.discover(root)]
            discovered[key] = {
                str(pair["recording_id"]): pair for pair in pairs
            }
            corpus_counts[str(item["dataset"])] = len(pairs)
        adapter = get_dataset_adapter(adapter_name)
        pair = discovered[key].get(str(item["recording_id"]))
        if pair is None:
            raise ValueError(f"recording is unavailable: {item['recording_id']}")
        recording = adapter.load(Path(pair["imu_path"]), Path(pair["vicon_path"]))
        time_s = np.asarray(recording.vicon_time_s, dtype=np.float64)
        position_m = np.asarray(recording.vicon_position_m, dtype=np.float64)
        edge_order = 2 if len(time_s) >= 3 else 1
        velocity_mps = np.gradient(position_m, time_s, axis=0, edge_order=edge_order)
        speed_mps = np.linalg.norm(velocity_mps, axis=1)
        if not np.all(np.isfinite(speed_mps)):
            raise ValueError(f"non-finite reference speed: {recording.id}")
        loaded.append(
            {
                **item,
                "time_s": time_s,
                "position_m": position_m,
                "speed_mps": speed_mps,
            }
        )
    return loaded, corpus_counts


def _decimated_indices(length: int) -> np.ndarray:
    if length <= MAX_PLOTTED_POINTS:
        return np.arange(length, dtype=np.int64)
    return np.unique(
        np.linspace(0, length - 1, MAX_PLOTTED_POINTS, dtype=np.int64)
    )


def _line_collection(
    position_xy: np.ndarray,
    display_speed: np.ndarray,
    norm: mpl.colors.Normalize,
) -> LineCollection:
    points = position_xy[:, None, :]
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    values = 0.5 * (display_speed[:-1] + display_speed[1:])
    collection = LineCollection(
        segments,
        cmap=CMAP,
        norm=norm,
        linewidth=0.72,
        alpha=0.96,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_array(values)
    return collection


def _square_limits(records: list[dict[str, object]]) -> dict[str, float]:
    by_dataset: dict[str, float] = {}
    for record in records:
        xy = np.asarray(record["position_m"])[:, :2]
        span = np.ptp(xy, axis=0)
        by_dataset[str(record["dataset"])] = max(
            by_dataset.get(str(record["dataset"]), 0.0),
            float(max(span)),
        )
    return {name: max(span * 1.10, 1.0) for name, span in by_dataset.items()}


def _source_rows(
    records: list[dict[str, object]], *, vmax: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for panel_index, record in enumerate(records):
        time_s = np.asarray(record["time_s"])
        position = np.asarray(record["position_m"])
        speed = np.asarray(record["speed_mps"])
        indices = _decimated_indices(len(time_s))
        for source_index in indices:
            rows.append(
                {
                    "panel": chr(ord("a") + panel_index),
                    "dataset": record["dataset"],
                    "domain": record["domain"],
                    "recording_id": record["recording_id"],
                    "source_index": int(source_index),
                    "time_s": float(time_s[source_index]),
                    "x_m": float(position[source_index, 0]),
                    "y_m": float(position[source_index, 1]),
                    "z_m": float(position[source_index, 2]),
                    "speed_mps": float(speed[source_index]),
                    "display_speed_mps": float(min(speed[source_index], vmax)),
                }
            )
        distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
        summaries.append(
            {
                "panel": chr(ord("a") + panel_index),
                "dataset": record["dataset"],
                "recording_id": record["recording_id"],
                "source_rows": len(time_s),
                "plotted_rows": len(indices),
                "duration_s": float(time_s[-1] - time_s[0]),
                "path_length_m": float(distance.sum()),
                "speed_median_mps": float(np.median(speed)),
                "speed_p95_mps": float(np.percentile(speed, 95.0)),
                "speed_max_mps": float(np.max(speed)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def create_figure(output_directory: Path) -> dict[str, object]:
    records, corpus_counts = _load_selected()
    # Column-major scientific logic: two representatives per dataset.
    ordered = [records[index] for index in (0, 2, 4, 1, 3, 5)]
    pooled_speed = np.concatenate(
        [np.asarray(record["speed_mps"]) for record in ordered]
    )
    vmax = float(np.percentile(pooled_speed, 98.5))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("unable to determine a positive shared speed scale")
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
    dataset_spans = _square_limits(ordered)

    width_in = FIGURE_WIDTH_MM / 25.4
    height_in = FIGURE_HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        left=0.075,
        right=0.90,
        bottom=0.115,
        top=0.90,
        wspace=0.28,
        hspace=0.30,
    )
    axes = np.array(
        [[fig.add_subplot(grid[row, column]) for column in range(3)] for row in range(2)]
    )
    cax = fig.add_subplot(grid[:, 3])

    column_headers = (
        ("OxIOD", "smartphone · n=45 recordings"),
        ("EuRoC MAV", "MAV · n=6 sequences"),
        ("IDOL", "pedestrian · n=130 trajectories"),
    )
    source_frames, summary_frame = _source_rows(ordered, vmax=vmax)
    for panel_index, (axis, record) in enumerate(zip(axes.flat, ordered)):
        position = np.asarray(record["position_m"])
        speed = np.asarray(record["speed_mps"])
        indices = _decimated_indices(len(position))
        xy = position[indices, :2]
        display_speed = np.minimum(speed[indices], vmax)
        axis.add_collection(_line_collection(xy, display_speed, norm))
        axis.scatter(
            xy[0, 0],
            xy[0, 1],
            s=18,
            marker="o",
            facecolor=START_COLOR,
            edgecolor="#1F1F1F",
            linewidth=0.55,
            zorder=4,
        )
        axis.scatter(
            xy[-1, 0],
            xy[-1, 1],
            s=22,
            marker="x",
            color=END_COLOR,
            linewidth=1.25,
            zorder=4,
        )
        extent = dataset_spans[str(record["dataset"])]
        center = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
        half = 0.5 * extent
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_aspect("equal", adjustable="box")
        axis.set_box_aspect(1)
        axis.set_title(str(record["label"]), fontsize=6.8, pad=3.0)
        axis.text(
            -0.13,
            1.035,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=8.3,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        axis.grid(color="#D9D9D9", linewidth=0.42, alpha=0.55)
        axis.tick_params(labelsize=5.5, pad=1.5)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        if panel_index < 3:
            dataset_name, detail = column_headers[panel_index]
            axis.text(
                0.5,
                1.21,
                dataset_name,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.1,
                fontweight="bold",
            )
            axis.text(
                0.5,
                1.12,
                detail,
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#606060",
            )

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    colorbar_position = cax.get_position()
    cax.set_position(
        [
            colorbar_position.x0,
            colorbar_position.y0 + 0.12,
            colorbar_position.width,
            colorbar_position.height - 0.12,
        ]
    )
    colorbar = fig.colorbar(scalar, cax=cax, extend="max")
    colorbar.set_label("Ground-truth speed (m s$^{-1}$)", fontsize=6.3, labelpad=4)
    colorbar.ax.tick_params(labelsize=5.3, width=0.5, length=2.2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

    marker_handles = (
        Line2D(
            [], [], marker="o", linestyle="none", markersize=4.6,
            markerfacecolor=START_COLOR, markeredgecolor="#1F1F1F", label="Start"
        ),
        Line2D(
            [], [], marker="x", linestyle="none", markersize=4.8,
            markeredgewidth=1.1, color=END_COLOR, label="End"
        ),
    )
    fig.legend(
        handles=marker_handles,
        loc="lower left",
        bbox_to_anchor=(0.89, 0.105),
        ncol=1,
        frameon=False,
        fontsize=5.8,
        handletextpad=0.35,
        labelspacing=0.45,
    )
    fig.supxlabel("X position (m)", x=0.49, y=0.045, fontsize=6.7)
    fig.supylabel("Y position (m)", x=0.018, y=0.51, fontsize=6.7)

    output_directory.mkdir(parents=True, exist_ok=True)
    base = output_directory / "ThreeDatasetTrajectories"
    outputs: list[str] = []
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", **options)
        outputs.append(str(path))
    plt.close(fig)

    source_path = output_directory / "SourceData.csv"
    summary_path = output_directory / "TrajectorySummary.csv"
    source_frames.to_csv(source_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    caption_path = output_directory / "FigureCaption.md"
    caption_path.write_text(
        "**Representative motion domains across the three IMU corpora.** "
        "Two ground-truth trajectories are shown for each dataset: smartphone "
        "motion in OxIOD (a,d), MAV flight in EuRoC (b,e), and indoor pedestrian "
        "motion in IDOL (c,f). Line colour denotes three-dimensional ground-truth "
        f"speed on a shared scale (display clipped above {vmax:.2f} m s^-1); "
        "green circles and red crosses mark trajectory start and end, respectively. "
        "Coordinates are shown in each dataset's native reference frame, with a "
        "common spatial scale within each dataset column. Trajectories were "
        "deterministically decimated for rendering only; no smoothing, spatial "
        "registration, or geometric rescaling was applied.\n",
        encoding="utf-8",
    )
    qa_path = output_directory / "QANotes.md"
    qa_path.write_text(
        "# Figure QA notes\n\n"
        "- Core conclusion: one pipeline spans smartphone, MAV, and pedestrian motion domains.\n"
        "- Archetype: quantitative trajectory grid (2 representatives × 3 datasets).\n"
        "- Final size: 183 × 116 mm (double-column).\n"
        "- Backend: Python/matplotlib only.\n"
        "- Primary export: SVG with editable text; PDF, 600-dpi TIFF, and 300-dpi PNG included.\n"
        "- Data integrity: native coordinates; no smoothing, registration, or spatial rescaling.\n"
        f"- Colour integrity: shared viridis scale, 0–{vmax:.3f} m/s; values above the pooled 98.5th percentile are colour-clipped only.\n"
        f"- Rendering: at most {MAX_PLOTTED_POINTS} deterministic points per trajectory; source indices are recorded.\n"
        "- Interpretation limit: representative dataset coverage, not an imputation-performance comparison.\n"
        "- Statistics: none; this figure contains representative trajectories rather than inferential estimates.\n",
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "outputs": outputs,
        "source_data": str(source_path),
        "summary": str(summary_path),
        "caption": str(caption_path),
        "qa_notes": str(qa_path),
        "speed_vmax_mps": vmax,
        "corpus_counts": corpus_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "figures" / "ThreeDatasetTrajectories",
    )
    args = parser.parse_args()
    print(json.dumps(create_figure(args.output_directory.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
