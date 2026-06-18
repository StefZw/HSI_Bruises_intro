from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
from IPython.display import display


def find_data_raw_root(start_path: Optional[str] = None) -> Path:
    """Find the Data/Raw folder by walking up from start_path (or cwd)."""
    base = Path(start_path).expanduser().resolve() if start_path else Path.cwd().resolve()
    candidates: Iterable[Path] = [base, *base.parents]
    for candidate in candidates:
        raw = candidate / "Data" / "Raw"
        if raw.exists() and raw.is_dir():
            return raw
    raise FileNotFoundError("Could not find Data/Raw from current location.")


def _capture_paths(folder_name: str, raw_root: Path) -> Tuple[Path, Path, Path, Optional[Path]]:
    folder = raw_root / folder_name
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    capture = folder / "capture"
    sample_hdr = capture / f"{folder_name}.hdr"
    dark_hdr = capture / f"DARKREF_{folder_name}.hdr"
    white_hdr = capture / f"WHITEREF_{folder_name}.hdr"

    missing = [p for p in [sample_hdr, dark_hdr, white_hdr] if not p.exists()]
    if missing:
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required capture files:\n{missing_text}")

    png_path = folder / f"{folder_name}.png"
    if not png_path.exists():
        png_path = None

    return sample_hdr, dark_hdr, white_hdr, png_path


def load_capture_triplet(folder_name: str, raw_root: Optional[str] = None):
    """Load sample, dark and white captures for one acquisition folder."""
    root = Path(raw_root).expanduser().resolve() if raw_root else find_data_raw_root()
    sample_hdr, dark_hdr, white_hdr, png_path = _capture_paths(folder_name, root)

    sample_data = spy.open_image(str(sample_hdr)).load()
    dark_data = spy.open_image(str(dark_hdr)).load()
    white_data = spy.open_image(str(white_hdr)).load()

    return sample_data, dark_data, white_data, png_path


def get_default_wavelengths(cube: np.ndarray, min_nm: float = 400.0, max_nm: float = 1000.0) -> np.ndarray:
    """Create a default wavelength axis based on number of bands."""
    return np.linspace(min_nm, max_nm, cube.shape[2])


def show_png_with_grid(png_path: Optional[str], grid_step: int = 50, figsize: Tuple[int, int] = (10, 6)):
    """Display the PNG overview image with a raster grid overlay."""
    if png_path is None:
        print("No PNG image found for this folder.")
        return None

    img = plt.imread(str(png_path))
    height, width = img.shape[:2]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.set_title(Path(png_path).name)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    for x in range(0, width + 1, grid_step):
        ax.axvline(x, color="white", alpha=0.35, linewidth=0.8)
    for y in range(0, height + 1, grid_step):
        ax.axhline(y, color="white", alpha=0.35, linewidth=0.8)

    ax.set_xticks(np.arange(0, width + 1, grid_step))
    ax.set_yticks(np.arange(0, height + 1, grid_step))
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    return fig, ax


def calibrate_data(sample_data: np.ndarray, dark_data: np.ndarray, white_data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return calibrated reflectance cube: (sample-dark)/(white-dark)."""
    sample_f = sample_data.astype(np.float32)
    dark_f = dark_data.astype(np.float32)
    white_f = white_data.astype(np.float32)
    return np.divide(sample_f - dark_f, white_f - dark_f + eps)


def plot_white_dark_spectrum(
    white_data: np.ndarray,
    dark_data: np.ndarray,
    wavelengths: np.ndarray,
    row: int = 0,
    col: int = 256,
    figsize: Tuple[int, int] = (12, 5),
):
    """Plot white and dark reference spectra for one pixel location."""
    col = int(np.clip(col, 0, white_data.shape[1] - 1))
    row = int(np.clip(row, 0, white_data.shape[0] - 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wavelengths, np.squeeze(white_data[row, col, :]), color="gray", linewidth=2, label="White")
    ax.plot(wavelengths, np.squeeze(dark_data[row, col, :]), color="black", linewidth=2, label="Dark")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Signal")
    ax.set_title(f"White/Dark reference at row={row}, col={col}")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


def crop_rotate_roi(
    calibrated_data: np.ndarray,
    y_slice: Tuple[int, int],
    x_slice: Tuple[int, int],
    rotation_k: int = 0,
) -> np.ndarray:
    """Crop and rotate calibrated data. rotation_k uses np.rot90 convention."""
    y0, y1 = y_slice
    x0, x1 = x_slice
    roi = calibrated_data[y0:y1, x0:x1, :]
    if rotation_k % 4 != 0:
        roi = np.rot90(roi, k=rotation_k, axes=(0, 1))
    return roi


def create_roi_crop_rotate_ui(
    calibrated_data: np.ndarray,
    wavelengths: Optional[np.ndarray] = None,
    initial_y: Tuple[int, int] = (0, 100),
    initial_x: Tuple[int, int] = (0, 100),
    initial_rotation_k: int = 0,
    rgb_bands: Tuple[int, int, int] = (29, 19, 9),
):
    """Interactive ROI crop + rotation tool returning a mutable state dict."""
    h, w, _ = calibrated_data.shape
    y0_init = int(np.clip(initial_y[0], 0, h - 1))
    y1_init = int(np.clip(initial_y[1], 1, h))
    x0_init = int(np.clip(initial_x[0], 0, w - 1))
    x1_init = int(np.clip(initial_x[1], 1, w))

    y0_slider = widgets.IntSlider(value=y0_init, min=0, max=h - 1, step=1, description="y0")
    y1_slider = widgets.IntSlider(value=max(y1_init, y0_init + 1), min=1, max=h, step=1, description="y1")
    x0_slider = widgets.IntSlider(value=x0_init, min=0, max=w - 1, step=1, description="x0")
    x1_slider = widgets.IntSlider(value=max(x1_init, x0_init + 1), min=1, max=w, step=1, description="x1")
    rotation_slider = widgets.IntSlider(value=initial_rotation_k % 4, min=0, max=3, step=1, description="rot90 k")

    output = widgets.Output()
    state = {"roi": None, "y_slice": (y0_init, y1_init), "x_slice": (x0_init, x1_init), "rotation_k": initial_rotation_k % 4}

    def _render(_=None):
        y0, y1 = sorted((y0_slider.value, y1_slider.value))
        x0, x1 = sorted((x0_slider.value, x1_slider.value))
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        if x1 <= x0:
            x1 = min(w, x0 + 1)

        roi = crop_rotate_roi(calibrated_data, (y0, y1), (x0, x1), rotation_k=rotation_slider.value)
        state["roi"] = roi
        state["y_slice"] = (y0, y1)
        state["x_slice"] = (x0, x1)
        state["rotation_k"] = rotation_slider.value

        with output:
            output.clear_output(wait=True)
            rgb = spy.get_rgb(roi, list(rgb_bands))
            plt.figure(figsize=(8, 5))
            plt.imshow(rgb)
            title = f"ROI shape: {roi.shape}"
            if wavelengths is not None and len(wavelengths) > max(rgb_bands):
                title += (
                    f" | RGB bands: {rgb_bands[0]} ({wavelengths[rgb_bands[0]]:.1f} nm),"
                    f" {rgb_bands[1]} ({wavelengths[rgb_bands[1]]:.1f} nm),"
                    f" {rgb_bands[2]} ({wavelengths[rgb_bands[2]]:.1f} nm)"
                )
            plt.title(title)
            plt.axis("off")
            plt.show()

    for widget in [y0_slider, y1_slider, x0_slider, x1_slider, rotation_slider]:
        widget.observe(_render, names="value")

    _render()
    ui = widgets.VBox([
        widgets.HBox([y0_slider, y1_slider]),
        widgets.HBox([x0_slider, x1_slider]),
        rotation_slider,
        output,
    ])
    display(ui)
    state["ui"] = ui
    return state


def visualize_two_point_spectra(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    point_a: Tuple[int, int],
    point_b: Tuple[int, int],
    rgb_bands: Tuple[int, int, int] = (29, 19, 9),
    labels: Tuple[str, str] = ("Point A", "Point B"),
    figsize: Tuple[int, int] = (12, 5),
):
    """Plot spectra for two points and show their locations on RGB composite."""
    ay, ax = point_a
    by, bx = point_b

    ay = int(np.clip(ay, 0, cube.shape[0] - 1))
    ax = int(np.clip(ax, 0, cube.shape[1] - 1))
    by = int(np.clip(by, 0, cube.shape[0] - 1))
    bx = int(np.clip(bx, 0, cube.shape[1] - 1))

    rgb = spy.get_rgb(cube, list(rgb_bands))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(wavelengths, np.squeeze(cube[ay, ax, :]), color="green", linewidth=2, label=f"{labels[0]} ({ay}, {ax})")
    axes[0].plot(wavelengths, np.squeeze(cube[by, bx, :]), color="red", linewidth=2, label=f"{labels[1]} ({by}, {bx})")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectance")
    axes[0].set_title("Two-point spectral signatures")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].imshow(rgb)
    axes[1].scatter(ax, ay, c="green", s=80, marker="o")
    axes[1].scatter(bx, by, c="red", s=80, marker="o")
    axes[1].set_title("RGB composite with selected points")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def create_band_slider_ui(cube: np.ndarray, wavelengths: np.ndarray, cmap: str = "gray"):
    """Interactive viewer for single-band images."""

    def _show_band(b: int):
        plt.figure(figsize=(10, 5))
        plt.imshow(cube[:, :, b], cmap=cmap)
        plt.title(f"Band {b} ({wavelengths[b]:.1f} nm)")
        plt.axis("off")
        plt.show()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=cube.shape[2] - 1,
        step=1,
        description="Band",
        continuous_update=True,
    )
    out = widgets.interactive_output(_show_band, {"b": slider})
    ui = widgets.VBox([slider, out])
    display(ui)
    return ui


def create_band_ratio_ui(cube: np.ndarray, wavelengths: np.ndarray, eps: float = 1e-8):
    """Interactive viewer for ratio map band_A / band_B."""

    def _show_ratio(a: int, b: int):
        ratio = cube[:, :, a] / (cube[:, :, b] + eps)
        vmin, vmax = np.percentile(ratio, [1, 99])

        plt.figure(figsize=(10, 5))
        plt.imshow(ratio, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title(f"Band ratio {a}/{b} ({wavelengths[a]:.1f} / {wavelengths[b]:.1f} nm)")
        plt.axis("off")
        plt.show()

    slider_a = widgets.IntSlider(
        value=min(55, cube.shape[2] - 1),
        min=0,
        max=cube.shape[2] - 1,
        step=1,
        description="Band A",
        continuous_update=True,
    )
    slider_b = widgets.IntSlider(
        value=min(26, cube.shape[2] - 1),
        min=0,
        max=cube.shape[2] - 1,
        step=1,
        description="Band B",
        continuous_update=True,
    )
    out = widgets.interactive_output(_show_ratio, {"a": slider_a, "b": slider_b})
    ui = widgets.VBox([slider_a, slider_b, out])
    display(ui)
    return ui


def _extract_mean_spectrum(cube: np.ndarray, row: int, col: int, radius: int = 3) -> np.ndarray:
    r0 = max(0, row - radius)
    r1 = min(cube.shape[0], row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(cube.shape[1], col + radius + 1)
    return cube[r0:r1, c0:c1, :].mean(axis=(0, 1))


def _spectral_angle_map(cube: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    h, w, b = cube.shape
    pixels = cube.reshape(-1, b).astype(np.float32)
    pixels_unit = pixels / (np.linalg.norm(pixels, axis=1, keepdims=True) + 1e-8)

    angles = []
    for spectrum in endmembers:
        s = spectrum.astype(np.float32)
        s = s / (np.linalg.norm(s) + 1e-8)
        cosang = np.clip(pixels_unit @ s, -1.0, 1.0)
        angles.append(np.arccos(cosang))

    return np.stack(angles, axis=1).reshape(h, w, -1)


def run_sam(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    reference_points: Dict[str, Tuple[int, int]],
    window_radius: int = 3,
    threshold_rad: Optional[float] = 0.12,
    rgb_bands: Tuple[int, int, int] = (29, 19, 9),
):
    """Run SAM and visualize reference spectra, point map, angle maps and class map."""
    class_names = list(reference_points.keys())
    reference_spectra = np.array(
        [_extract_mean_spectrum(cube, r, c, radius=window_radius) for r, c in reference_points.values()]
    )

    sam_angles = _spectral_angle_map(cube, reference_spectra)
    sam_min_angle = np.min(sam_angles, axis=2)
    sam_class_map = np.argmin(sam_angles, axis=2)

    if threshold_rad is not None:
        sam_class_map_masked = sam_class_map.astype(float)
        sam_class_map_masked[sam_min_angle > threshold_rad] = np.nan
    else:
        sam_class_map_masked = sam_class_map

    rgb = spy.get_rgb(cube, list(rgb_bands))

    plt.figure(figsize=(10, 5))
    for i, name in enumerate(class_names):
        plt.plot(wavelengths, reference_spectra[i], linewidth=2, label=name)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Reference spectra for SAM")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.imshow(rgb)
    for name, (row, col) in reference_points.items():
        plt.scatter(col, row, s=100, edgecolor="white", linewidth=1.5, label=name)
    plt.title("Reference points for SAM")
    plt.axis("off")
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5))
    if len(class_names) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = sam_angles[:, :, i]
        vmin_i, vmax_i = np.percentile(img, [1, 99])
        im = ax.imshow(img, cmap="viridis", vmin=vmin_i, vmax=vmax_i)
        ax.set_title(f"SAM angle: {class_names[i]}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(
        sam_class_map_masked,
        cmap=plt.cm.get_cmap("Set1", len(class_names)),
        vmin=0,
        vmax=len(class_names) - 1,
    )
    ax.set_title("SAM classification")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(class_names)), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(class_names)
    plt.show()

    return {
        "class_names": class_names,
        "reference_spectra": reference_spectra,
        "sam_angles": sam_angles,
        "sam_min_angle": sam_min_angle,
        "sam_class_map": sam_class_map,
        "sam_class_map_masked": sam_class_map_masked,
    }
