import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import brainmass
import brainunit as u
import matplotlib
import numpy as np
from scipy import optimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")


def quantity_to_float(value) -> float:
    """Convert a brainunit quantity or array-like value to a Python float."""
    if hasattr(value, "mantissa"):
        array = np.asarray(value.mantissa, dtype=float)
    else:
        array = np.asarray(value, dtype=float)
    return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class FixedPoint:
    ext_e: float
    ext_i: float
    e: float
    i: float
    stable: bool
    eigvals: tuple[complex, complex]


class BrainmassWilsonCowan:
    """A thin wrapper that keeps the Wilson-Cowan equations tied to brainmass."""

    def __init__(self, **model_kwargs):
        self.model = brainmass.WilsonCowanStep(1, **model_kwargs)
        self.model_kwargs = model_kwargs

    def derivative(self, e: float, i: float, ext_e: float = 0.0, ext_i: float = 0.0) -> np.ndarray:
        de = quantity_to_float(self.model.drE(e, i, ext_e))
        di = quantity_to_float(self.model.drI(i, e, ext_i))
        return np.array([de, di], dtype=float)


def rk4_integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    state0: Iterable[float],
    duration_ms: float,
    dt_ms: float,
    clip_max: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(0.0, duration_ms + dt_ms, dt_ms)
    states = np.zeros((time.size, len(tuple(state0))), dtype=float)
    states[0] = np.asarray(tuple(state0), dtype=float)

    for k in range(time.size - 1):
        t = time[k]
        y = states[k]
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt_ms, y + 0.5 * dt_ms * k1)
        k3 = rhs(t + 0.5 * dt_ms, y + 0.5 * dt_ms * k2)
        k4 = rhs(t + dt_ms, y + dt_ms * k3)
        y_next = y + (dt_ms / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states[k + 1] = np.clip(y_next, 0.0, clip_max)

    return time, states


def simulate_single_node(
    wc: BrainmassWilsonCowan,
    duration_ms: float,
    dt_ms: float,
    ext_e_fn: Callable[[float], float],
    ext_i_fn: Callable[[float], float] | None = None,
    state0: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ext_i_fn = ext_i_fn or (lambda _t: 0.0)

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        e, i = state
        return wc.derivative(e, i, ext_e_fn(t), ext_i_fn(t))

    time, states = rk4_integrate(rhs, state0, duration_ms, dt_ms)
    ext_e = np.array([ext_e_fn(t) for t in time], dtype=float)
    return time, states[:, 0], states[:, 1], ext_e


def simulate_decision_network(
    left_wc: BrainmassWilsonCowan,
    right_wc: BrainmassWilsonCowan,
    duration_ms: float,
    dt_ms: float,
    cue_on_ms: float = 50.0,
    cue_off_ms: float = 220.0,
    baseline_drive: float = 0.30,
    evidence_drive: float = 1.10,
    coherence: float = 0.15,
    competition: float = 2.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def cue(t: float) -> float:
        return 1.0 if cue_on_ms <= t <= cue_off_ms else 0.0

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        e_left, i_left, e_right, i_right = state
        stimulus_left = baseline_drive + cue(t) * (evidence_drive + coherence)
        stimulus_right = baseline_drive + cue(t) * (evidence_drive - coherence)
        left = left_wc.derivative(e_left, i_left, stimulus_left - competition * e_right, 0.0)
        right = right_wc.derivative(e_right, i_right, stimulus_right - competition * e_left, 0.0)
        return np.array([left[0], left[1], right[0], right[1]], dtype=float)

    time, states = rk4_integrate(rhs, (0.0, 0.0, 0.0, 0.0), duration_ms, dt_ms)
    inputs = np.column_stack(
        [
            baseline_drive + np.array([cue(t) * (evidence_drive + coherence) for t in time], dtype=float),
            baseline_drive + np.array([cue(t) * (evidence_drive - coherence) for t in time], dtype=float),
        ]
    )
    return time, states[:, 0], states[:, 1], states[:, 2], inputs


def finite_difference_jacobian(
    wc: BrainmassWilsonCowan,
    e: float,
    i: float,
    ext_e: float,
    ext_i: float,
    delta: float = 1e-5,
) -> np.ndarray:
    center = np.array([e, i], dtype=float)
    jacobian = np.zeros((2, 2), dtype=float)

    for axis in range(2):
        plus = center.copy()
        minus = center.copy()
        plus[axis] += delta
        minus[axis] -= delta
        jacobian[:, axis] = (
            wc.derivative(plus[0], plus[1], ext_e, ext_i)
            - wc.derivative(minus[0], minus[1], ext_e, ext_i)
        ) / (2.0 * delta)

    return jacobian


def find_fixed_points(
    wc: BrainmassWilsonCowan,
    ext_e: float = 0.0,
    ext_i: float = 0.0,
    n_seeds: int = 7,
    merge_tol: float = 1e-3,
) -> list[FixedPoint]:
    fixed_points: list[FixedPoint] = []

    def rhs(state: np.ndarray) -> np.ndarray:
        return wc.derivative(state[0], state[1], ext_e, ext_i)

    seeds = np.linspace(0.01, 0.99, n_seeds)
    for e0 in seeds:
        for i0 in seeds:
            result = optimize.root(rhs, np.array([e0, i0], dtype=float), method="hybr")
            if not result.success:
                continue

            e_star, i_star = result.x
            if not (-1e-4 <= e_star <= 1.0 and -1e-4 <= i_star <= 1.0):
                continue

            candidate = np.array([e_star, i_star], dtype=float)
            if any(np.linalg.norm(candidate - np.array([fp.e, fp.i])) < merge_tol for fp in fixed_points):
                continue

            jacobian = finite_difference_jacobian(wc, candidate[0], candidate[1], ext_e, ext_i)
            eigvals = tuple(np.linalg.eigvals(jacobian))
            fixed_points.append(
                FixedPoint(
                    ext_e=ext_e,
                    ext_i=ext_i,
                    e=float(candidate[0]),
                    i=float(candidate[1]),
                    stable=bool(np.all(np.real(eigvals) < 0.0)),
                    eigvals=eigvals,
                )
            )

    fixed_points.sort(key=lambda fp: fp.e)
    return fixed_points


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_phase_plane_and_bifurcation(output_dir: Path, dpi: int) -> None:
    wc = BrainmassWilsonCowan()
    ext_e = 0.0
    ext_i = 0.0

    e_grid = np.linspace(0.0, 0.60, 180)
    i_grid = np.linspace(0.0, 0.35, 180)
    ee, ii = np.meshgrid(e_grid, i_grid)
    de = np.zeros_like(ee)
    di = np.zeros_like(ii)

    for row in range(ee.shape[0]):
        for col in range(ee.shape[1]):
            derivative = wc.derivative(ee[row, col], ii[row, col], ext_e, ext_i)
            de[row, col] = derivative[0]
            di[row, col] = derivative[1]

    speed = np.sqrt(de**2 + di**2) + 1e-9
    fixed_points = find_fixed_points(wc, ext_e=ext_e, ext_i=ext_i)

    ext_values = np.linspace(-0.02, 0.35, 70)
    bifurcation_rows: list[dict] = []
    for current_ext in ext_values:
        for fp in find_fixed_points(wc, ext_e=float(current_ext), ext_i=0.0, n_seeds=6):
            bifurcation_rows.append(
                {
                    "ext_e": fp.ext_e,
                    "ext_i": fp.ext_i,
                    "e": fp.e,
                    "i": fp.i,
                    "stable": fp.stable,
                    "eigval_real_1": float(np.real(fp.eigvals[0])),
                    "eigval_imag_1": float(np.imag(fp.eigvals[0])),
                    "eigval_real_2": float(np.real(fp.eigvals[1])),
                    "eigval_imag_2": float(np.imag(fp.eigvals[1])),
                }
            )

    save_csv(
        output_dir / "bifurcation_points.csv",
        bifurcation_rows,
        [
            "ext_e",
            "ext_i",
            "e",
            "i",
            "stable",
            "eigval_real_1",
            "eigval_imag_1",
            "eigval_real_2",
            "eigval_imag_2",
        ],
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].streamplot(
        e_grid,
        i_grid,
        de / speed,
        di / speed,
        color=np.log10(speed),
        cmap="viridis",
        density=1.2,
        linewidth=1.0,
        arrowsize=0.8,
    )
    axes[0].contour(ee, ii, de, levels=[0.0], colors=["#d62728"], linewidths=2.5)
    axes[0].contour(ee, ii, di, levels=[0.0], colors=["#1f77b4"], linewidths=2.5)
    for fp in fixed_points:
        axes[0].scatter(
            fp.e,
            fp.i,
            s=95,
            c="#111111" if fp.stable else "white",
            edgecolors="#111111",
            linewidths=1.4,
            zorder=5,
        )
    axes[0].set_title("Phase Plane at Zero External Input")
    axes[0].set_xlabel("Excitatory activity E")
    axes[0].set_ylabel("Inhibitory activity I")
    axes[0].set_xlim(0.0, 0.60)
    axes[0].set_ylim(0.0, 0.35)
    axes[0].text(0.03, 0.31, "red: dE/dt = 0\nblue: dI/dt = 0", fontsize=10)

    stable_points = [row for row in bifurcation_rows if row["stable"]]
    unstable_points = [row for row in bifurcation_rows if not row["stable"]]
    axes[1].scatter(
        [row["ext_e"] for row in stable_points],
        [row["e"] for row in stable_points],
        s=18,
        color="#1f77b4",
        label="stable fixed points",
    )
    axes[1].scatter(
        [row["ext_e"] for row in unstable_points],
        [row["e"] for row in unstable_points],
        s=18,
        color="#ff7f0e",
        label="unstable fixed points",
    )
    axes[1].set_title("Bifurcation Scan Along External Excitation")
    axes[1].set_xlabel("External excitatory drive")
    axes[1].set_ylabel("Fixed-point E activity")
    axes[1].legend(frameon=True)

    fig.suptitle("Wilson-Cowan Bistability", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "phase_plane_and_bifurcation.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sensory_representation(output_dir: Path, dpi: int, dt_ms: float) -> None:
    wc = BrainmassWilsonCowan(wEE=10.0, wIE=12.0, wEI=13.0, wII=11.0)

    def stimulus(t_ms: float) -> float:
        return 3.2 if 50.0 <= t_ms <= 120.0 else 0.0

    time, e, i, ext_e = simulate_single_node(wc, duration_ms=250.0, dt_ms=dt_ms, ext_e_fn=stimulus)
    save_csv(
        output_dir / "sensory_representation.csv",
        [
            {"time_ms": t, "ext_e": inp, "E": exc, "I": inh}
            for t, inp, exc, inh in zip(time, ext_e, e, i)
        ],
        ["time_ms", "ext_e", "E", "I"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[1, 2])
    axes[0].fill_between(time, 0.0, ext_e, color="#cfe2f3")
    axes[0].plot(time, ext_e, color="#1f77b4", linewidth=2.2)
    axes[0].set_ylabel("Input")
    axes[0].set_title("Sensory Representation: transient coding without persistence")

    axes[1].plot(time, e, color="#d62728", linewidth=2.4, label="E")
    axes[1].plot(time, i, color="#2ca02c", linewidth=2.0, label="I")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Activity")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "sensory_representation.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_working_memory(output_dir: Path, dpi: int, dt_ms: float) -> None:
    wc = BrainmassWilsonCowan()

    def stimulus(t_ms: float) -> float:
        return 3.8 if 80.0 <= t_ms <= 180.0 else 0.15

    time, e, i, ext_e = simulate_single_node(wc, duration_ms=400.0, dt_ms=dt_ms, ext_e_fn=stimulus)
    fixed_points = find_fixed_points(wc, ext_e=0.15, ext_i=0.0)
    save_csv(
        output_dir / "working_memory.csv",
        [
            {"time_ms": t, "ext_e": inp, "E": exc, "I": inh}
            for t, inp, exc, inh in zip(time, ext_e, e, i)
        ],
        ["time_ms", "ext_e", "E", "I"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1, 2])
    axes[0].fill_between(time, 0.0, ext_e, color="#f7d7c4")
    axes[0].plot(time, ext_e, color="#ff7f0e", linewidth=2.2)
    axes[0].set_ylabel("Input")
    axes[0].set_title("Working Memory: pulse-driven transition into a persistent attractor")

    axes[1].plot(time, e, color="#d62728", linewidth=2.5, label="E")
    axes[1].plot(time, i, color="#2ca02c", linewidth=2.0, label="I")
    for fp in fixed_points:
        if fp.stable:
            axes[1].axhline(fp.e, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.7)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Activity")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "working_memory.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_perceptual_decision(output_dir: Path, dpi: int, dt_ms: float) -> None:
    left_wc = BrainmassWilsonCowan(wEE=13.5, wIE=11.5, wEI=13.0, wII=11.0)
    right_wc = BrainmassWilsonCowan(wEE=13.5, wIE=11.5, wEI=13.0, wII=11.0)
    time, e_left, i_left, e_right, inputs = simulate_decision_network(
        left_wc,
        right_wc,
        duration_ms=350.0,
        dt_ms=dt_ms,
    )
    save_csv(
        output_dir / "perceptual_decision.csv",
        [
            {
                "time_ms": t,
                "input_left": inp_left,
                "input_right": inp_right,
                "E_left": exc_left,
                "I_left": inh_left,
                "E_right": exc_right,
            }
            for t, inp_left, inp_right, exc_left, inh_left, exc_right in zip(
                time,
                inputs[:, 0],
                inputs[:, 1],
                e_left,
                i_left,
                e_right,
            )
        ],
        ["time_ms", "input_left", "input_right", "E_left", "I_left", "E_right"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True, height_ratios=[1, 2])
    axes[0].plot(time, inputs[:, 0], color="#1f77b4", linewidth=2.2, label="left evidence")
    axes[0].plot(time, inputs[:, 1], color="#9467bd", linewidth=2.2, label="right evidence")
    axes[0].set_ylabel("Input")
    axes[0].set_title("Perceptual Decision: weak evidence bias amplified by competition")
    axes[0].legend(frameon=True)

    axes[1].plot(time, e_left, color="#d62728", linewidth=2.5, label="left choice population")
    axes[1].plot(time, e_right, color="#17becf", linewidth=2.5, label="right choice population")
    axes[1].plot(time, e_left - e_right, color="#111111", linewidth=1.8, linestyle="--", label="decision variable")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Activity")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "perceptual_decision.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_motor_control(output_dir: Path, dpi: int, dt_ms: float) -> None:
    wc = BrainmassWilsonCowan(tau_I=2.5 * u.ms, wEE=13.0, wIE=12.0, wEI=13.0, wII=11.0)

    def ramp_drive(t_ms: float) -> float:
        ramp = np.clip((t_ms - 60.0) / 240.0, 0.0, 1.0)
        return 0.35 + 2.2 * ramp

    time, e, i, ext_e = simulate_single_node(wc, duration_ms=500.0, dt_ms=dt_ms, ext_e_fn=ramp_drive)
    threshold = 0.45
    onset_index = np.flatnonzero(e >= threshold)
    onset_time = time[onset_index[0]] if onset_index.size else np.nan

    save_csv(
        output_dir / "motor_control.csv",
        [
            {"time_ms": t, "ext_e": inp, "E": exc, "I": inh}
            for t, inp, exc, inh in zip(time, ext_e, e, i)
        ],
        ["time_ms", "ext_e", "E", "I"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1, 2])
    axes[0].plot(time, ext_e, color="#8c564b", linewidth=2.4)
    axes[0].fill_between(time, 0.0, ext_e, color="#ead7cf")
    axes[0].set_ylabel("Input")
    axes[0].set_title("Motor Control: ramp-to-threshold initiation")

    axes[1].plot(time, e, color="#d62728", linewidth=2.5, label="E")
    axes[1].plot(time, i, color="#2ca02c", linewidth=2.0, label="I")
    axes[1].axhline(threshold, color="#111111", linestyle="--", linewidth=1.4, label="movement threshold")
    if not np.isnan(onset_time):
        axes[1].axvline(onset_time, color="#111111", linestyle=":", linewidth=1.5)
        axes[1].text(onset_time + 6.0, threshold + 0.02, f"onset = {onset_time:.1f} ms", fontsize=10)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Activity")
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "motor_control.svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Wilson-Cowan analysis figures for PPT demos using brainmass."
    )
    parser.add_argument(
        "--demo",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "phase",
            "sensory",
            "working_memory",
            "decision",
            "motor",
        ],
        help="Which demo figures to generate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "wilson_cowan_outputs"),
        help="Directory for svg and CSV outputs.",
    )
    parser.add_argument("--dt-ms", type=float, default=0.1, help="Integration step size in ms.")
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.demo)
    if "all" in requested:
        requested = {"phase", "sensory", "working_memory", "decision", "motor"}

    if "phase" in requested:
        plot_phase_plane_and_bifurcation(output_dir, dpi=args.dpi)
    if "sensory" in requested:
        plot_sensory_representation(output_dir, dpi=args.dpi, dt_ms=args.dt_ms)
    if "working_memory" in requested:
        plot_working_memory(output_dir, dpi=args.dpi, dt_ms=args.dt_ms)
    if "decision" in requested:
        plot_perceptual_decision(output_dir, dpi=args.dpi, dt_ms=args.dt_ms)
    if "motor" in requested:
        plot_motor_control(output_dir, dpi=args.dpi, dt_ms=args.dt_ms)

    print("Generated outputs:")
    for path in sorted(output_dir.iterdir()):
        print(f" - {path}")


if __name__ == "__main__":
    main()
