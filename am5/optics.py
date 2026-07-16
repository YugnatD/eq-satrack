"""Rough exposure/gain estimation from the optical train + a selected pass.

Deliberately approximate throughout -- a photon-budget order-of-magnitude
estimate meant to give a starting point to dial in against the live preview
histogram, not a calibrated exposure calculator. Every constant below is a
commonly-used ballpark, not a rigorously derived value for a specific
camera/filter combination -- flagged individually below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARCSEC_PER_RADIAN = 206264.8


@dataclass(frozen=True)
class OpticalTrain:
    aperture_mm: float
    focal_length_mm: float
    barlow_multiplier: float
    pixel_size_um: float

    def __post_init__(self) -> None:
        # Regression fix: a fat-fingered "0" in the focal-length or pixel-
        # size field parses cleanly as a float (no ValueError), so callers
        # that only guard the float(...) parsing (get_optical_train,
        # ExposurePanel._on_compute_click, both in am5/gui/panels.py) went
        # on to divide by an effective focal length of zero inside
        # plate_scale_arcsec_per_px -- an uncaught ZeroDivisionError,
        # confirmed to leave ConnectionPanel's real-camera connect button
        # stuck disabled at "Connecting..." with no error shown and no way
        # to retry short of restarting the app. Raising here (a plain
        # ValueError, the same exception type every existing call site
        # already catches around its own float(...) parsing) turns the
        # crash into the same "invalid input" message those call sites
        # already show for a non-numeric field.
        if self.focal_length_mm <= 0 or self.barlow_multiplier <= 0:
            raise ValueError(
                f"effective focal length must be positive (focal_length_mm={self.focal_length_mm}, "
                f"barlow_multiplier={self.barlow_multiplier})"
            )
        if self.pixel_size_um <= 0:
            raise ValueError(f"pixel_size_um must be positive (got {self.pixel_size_um})")

    @property
    def effective_focal_length_mm(self) -> float:
        return self.focal_length_mm * self.barlow_multiplier

    @property
    def plate_scale_arcsec_per_px(self) -> float:
        return ARCSEC_PER_RADIAN * (self.pixel_size_um / 1000.0) / self.effective_focal_length_mm


def max_exposure_s(train: OpticalTrain, angular_speed_deg_s: float, max_trail_px: float = 1.0) -> float:
    """Longest exposure keeping the ISS's own motion blur under
    `max_trail_px` pixels, given its on-sky angular speed (deg/s, from
    Trajectory.sky_speed_deg_s()). This -- not sensor sensitivity -- is
    what actually caps exposure for a fast target like the ISS."""
    if angular_speed_deg_s <= 0:
        return float("inf")
    trail_budget_arcsec = train.plate_scale_arcsec_per_px * max_trail_px
    return trail_budget_arcsec / (angular_speed_deg_s * 3600.0)


def estimate_iss_magnitude(distance_km: float, ref_magnitude: float = -1.8, ref_distance_km: float = 1000.0) -> float:
    """Distance-scaled estimate from a commonly-cited ballpark brightness
    (~-1.8) at 1000km slant range. Ignores phase angle/illumination
    fraction, which real ISS brightness varies with by a magnitude or
    more -- an order-of-magnitude estimate, not a prediction to trust
    precisely."""
    if distance_km <= 0:
        return ref_magnitude
    return ref_magnitude + 5.0 * math.log10(distance_km / ref_distance_km)


# Broadband photon flux for a mag-0 source, photons/s/cm^2. The original
# value here (1e6, a generic "commonly cited" amateur-calculator ballpark)
# was off by ~4 orders of magnitude -- it predicted gain 0 for a real
# 200mm/f5 setup that empirically needed gain 220 (0.1dB units) at 1ms for
# a mag -3.3 ISS pass. Recalibrated by solving backwards from that real,
# verified capture (200mm aperture, 1000mm FL, no barlow, 1ms exposure,
# gain 220, mag -3.3) against DEFAULT_FULL_WELL_ELECTRONS and
# suggest_gain's target_fraction=0.6 -- an empirical anchor, not a
# textbook value, but a real one beats a guess (see am5/optics.py's
# module docstring philosophy).
MAG0_PHOTON_FLUX_PER_CM2_S = 135.5


def estimate_signal_electrons(train: OpticalTrain, magnitude: float, exposure_s: float, quantum_efficiency: float = 0.75) -> float:
    """quantum_efficiency default (~0.75) is the ballpark peak QE commonly
    quoted for the ASI290MC's sensor, not a per-wavelength curve."""
    aperture_area_cm2 = math.pi * (train.aperture_mm / 10.0 / 2.0) ** 2
    photon_rate = MAG0_PHOTON_FLUX_PER_CM2_S * 10.0 ** (-0.4 * magnitude) * aperture_area_cm2
    return photon_rate * exposure_s * quantum_efficiency


# Typical unity-gain full well for the ASI290MC, another ballpark (not this
# specific sensor's datasheet curve) -- shared so callers can also report
# how saturated the estimated signal already is at gain 0 (a suggested gain
# of 0 is ambiguous otherwise: "just enough" and "wildly oversaturated"
# look identical without this).
DEFAULT_FULL_WELL_ELECTRONS = 14000.0


def suggest_gain(signal_electrons: float, full_well_electrons: float = DEFAULT_FULL_WELL_ELECTRONS, target_fraction: float = 0.6) -> float:
    """Suggested ASI-style gain value (0-570, 0.1dB/step) to bring the
    estimated signal to `target_fraction` of a typical unity-gain full well
    (~14000e-, another ballpark, not this specific sensor's datasheet
    curve). A starting point -- check the live histogram and adjust,
    exactly like the rest of this module."""
    if signal_electrons <= 0 or full_well_electrons * target_fraction <= signal_electrons:
        return 0.0
    needed_multiplier = (full_well_electrons * target_fraction) / signal_electrons
    gain_db = 20.0 * math.log10(needed_multiplier)
    return max(0.0, min(570.0, gain_db * 10.0))


# Real-world span, in meters, that assets/iss_reference.jpg's full cropped
# width corresponds to -- the photo's solar arrays fill the frame edge to
# edge (verified: brightness at the outermost columns drops to near-zero,
# no stray bright pixels beyond the panels), so this is also the
# calibration used to derive the photo's own meters/pixel in
# render_iss_photo(). See assets/iss_reference.LICENSE.txt for provenance.
ISS_SOLAR_ARRAY_SPAN_M = 109.0

DEFAULT_ISS_REFERENCE_PHOTO = Path(__file__).resolve().parent.parent / "assets" / "iss_reference.jpg"

# Real optics never render a hard-edged silhouette -- seeing and the
# telescope's own diffraction limit blur it. Both are ballparks (typical
# amateur-site seeing; Dawes' rule for the diffraction limit), combined by
# taking whichever is larger, matching how these two blur sources are
# usually reasoned about informally (whichever dominates sets the blur).
TYPICAL_SEEING_FWHM_ARCSEC = 2.5
GAUSSIAN_FWHM_TO_SIGMA = 2.3548


def _diffraction_limit_arcsec(aperture_mm: float) -> float:
    return 116.0 / aperture_mm  # Dawes' limit, arcsec


def _gaussian_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur via plain numpy convolution -- no scipy
    dependency for one blur. Kernel radius is capped to the array size:
    np.convolve's 'same' mode returns len(kernel) (not len(row)) when the
    kernel is longer than the row, which silently corrupts the reshape
    downstream for a very fine plate scale (huge sigma in pixels)."""
    if sigma_px <= 0.05:
        return img
    max_radius = max(1, min(img.shape) // 2 - 1)
    radius = max(1, min(int(math.ceil(3.0 * sigma_px)), max_radius))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2.0 * sigma_px**2))
    kernel /= kernel.sum()
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=img)
    blurred = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=blurred)
    return blurred


DEFAULT_PREVIEW_FRAME_PX = 360


@dataclass(frozen=True)
class PixelationPreview:
    image: np.ndarray  # uint8 grayscale, always frame_px x frame_px (constant size, see render_iss_photo)
    angular_size_arcsec: float  # solar array span, at this distance
    camera_px_span: float  # true (unclamped) solar array span in actual camera pixels
    truncated: bool  # True if camera_px_span was too large to render at native scale


def render_iss_photo(
    train: OpticalTrain, distance_km: float, reference_path: Path = DEFAULT_ISS_REFERENCE_PHOTO,
    frame_px: int = DEFAULT_PREVIEW_FRAME_PX, max_native_px: int = 2000,
) -> PixelationPreview:
    """Resamples the real NASA reference photo (assets/iss_reference.jpg,
    grayscale) to the camera's actual pixel scale at this distance, blurs by
    seeing/diffraction, then fits the result into a *fixed* `frame_px` x
    `frame_px` display frame -- block-upscaled (nearest neighbor, stays
    visibly blocky) if undersampled, shrunk-to-fit if it overflows. The
    returned image size never changes with the optical train (only how
    coarse/fine the blocks within it look), so switching barlow multipliers
    doesn't resize the GUI widget, only what's shown inside it. Lazy-imports
    Pillow: only this function in the project needs a real image
    decoder/resampler."""
    from PIL import Image

    with Image.open(reference_path) as img:
        img = img.convert("L")
        ref_w, ref_h = img.size

        meters_per_px_ref = ISS_SOLAR_ARRAY_SPAN_M / ref_w
        target_meters_per_px = train.plate_scale_arcsec_per_px * distance_km * 1000.0 / ARCSEC_PER_RADIAN
        resize_ratio = meters_per_px_ref / target_meters_per_px  # true camera-pixel scale factor
        camera_px_w = ref_w * resize_ratio

        native_ratio = resize_ratio
        truncated = camera_px_w > max_native_px
        if truncated:
            native_ratio = max_native_px / ref_w
        native_w = max(1, round(ref_w * native_ratio))
        native_h = max(1, round(ref_h * native_ratio))

        native = img.resize((native_w, native_h), Image.LANCZOS)
        arr = np.asarray(native, dtype=np.float32) / 255.0

    if not truncated:
        # blur only makes sense at (near) the true camera-pixel scale --
        # skip it on a truncated (already heavily downscaled) render, where
        # LANCZOS's own anti-aliasing already dominates.
        blur_fwhm_arcsec = max(TYPICAL_SEEING_FWHM_ARCSEC, _diffraction_limit_arcsec(train.aperture_mm))
        blur_sigma_px = (blur_fwhm_arcsec / train.plate_scale_arcsec_per_px) / GAUSSIAN_FWHM_TO_SIGMA
        arr = _gaussian_blur(arr, blur_sigma_px)
    native_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

    fit_scale = frame_px / max(native_w, native_h)
    if fit_scale >= 1.0:
        block = max(1, round(fit_scale))
        fitted = np.kron(native_u8, np.ones((block, block), dtype=np.uint8))
    else:
        fitted = np.asarray(Image.fromarray(native_u8).resize(
            (max(1, round(native_w * fit_scale)), max(1, round(native_h * fit_scale))), Image.LANCZOS
        ))

    canvas = np.zeros((frame_px, frame_px), dtype=np.uint8)
    fh, fw = min(fitted.shape[0], frame_px), min(fitted.shape[1], frame_px)
    y0, x0 = (frame_px - fh) // 2, (frame_px - fw) // 2
    canvas[y0:y0 + fh, x0:x0 + fw] = fitted[:fh, :fw]

    angular_size_arcsec = ISS_SOLAR_ARRAY_SPAN_M / (distance_km * 1000.0) * ARCSEC_PER_RADIAN
    return PixelationPreview(image=canvas, angular_size_arcsec=angular_size_arcsec, camera_px_span=camera_px_w, truncated=truncated)
