"""Image stitching for videos with stable one-direction movement."""

import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


class StitchMode(enum.Enum):
    AUTO = "自动拼接"
    SEQUENTIAL = "顺序拼接"
    STRIP = "条带拼接"


class StitchDirection(enum.Enum):
    AUTO = "自动"
    HORIZONTAL = "水平"
    VERTICAL = "垂直"


@dataclass
class StitchResult:
    success: bool
    image: Optional[np.ndarray]
    message: str
    method_used: str = ""
    is_likely_single_frame: bool = False


@dataclass
class MotionInfo:
    direction: str
    avg_dx: float
    avg_dy: float
    avg_shift: float
    estimated_width: int
    estimated_height: int
    total_shift_x: float
    total_shift_y: float


def _resize_image(
    frame: np.ndarray, scale: float, high_quality: bool = False
) -> np.ndarray:
    """Resize a frame for preview/working resolution."""
    if scale >= 1.0 or scale <= 0:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_LANCZOS4 if high_quality else cv2.INTER_AREA
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def _resize_for_save(
    frame: np.ndarray, scale: float, high_quality: bool = False
) -> np.ndarray:
    """Resize for output, supporting upscaling."""
    if abs(scale - 1.0) < 0.01 or scale <= 0:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    if high_quality:
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def _estimate_translation(
    ref: np.ndarray, curr: np.ndarray
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Estimate pure translation using phase correlation."""
    try:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        (dx, dy), response = cv2.phaseCorrelate(
            np.float32(ref_gray), np.float32(curr_gray)
        )
        return (dx, dy), response
    except Exception:
        return None, 0.0


def _estimate_transform_orb(
    ref: np.ndarray, curr: np.ndarray, max_features: int = 500
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate affine transform using ORB feature matching + RANSAC.

    Returns a 2x3 affine matrix mapping points from ``curr`` to ``ref``.
    This is more robust than phase correlation on repetitive textures and
    sub-pixel shifts.
    """
    try:
        orb = cv2.ORB_create(max_features)
        kp1, des1 = orb.detectAndCompute(ref, None)
        kp2, des2 = orb.detectAndCompute(curr, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            return None, 0.0

        # Use top matches to reduce outliers.
        good = matches[: max(8, int(len(matches) * 0.3))]
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None:
            return None, 0.0

        inlier_ratio = float(np.sum(mask)) / len(mask) if mask is not None else 0.0
        return M.astype(np.float32), inlier_ratio
    except Exception:
        return None, 0.0


def _estimate_affine_robust(
    ref: np.ndarray, curr: np.ndarray, max_features: int = 800
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate affine transform using ORB + RANSAC, then refine with ECC.

    Returns a 2x3 affine matrix mapping points from ``curr`` to ``ref``.
    This handles translation, rotation, and scale changes, and is robust to
    repetitive textures and sub-pixel shifts.
    """
    # Step 1: ORB + RANSAC for initial affine estimate.
    matrix, inlier_ratio = _estimate_transform_orb(ref, curr, max_features)

    if matrix is None:
        # Fallback: phase correlation for translation only.
        shift, response = _estimate_translation(ref, curr)
        if shift is None:
            return None, 0.0
        matrix = _translation_matrix(shift[0], shift[1])
        inlier_ratio = response

    # Step 2: Refine with ECC using MOTION_AFFINE for sub-pixel accuracy.
    try:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        warp_matrix = matrix.copy().astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.0001)
        cc, warp_matrix = cv2.findTransformECC(
            ref_gray,
            curr_gray,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            3,
        )
        return warp_matrix.astype(np.float32), max(cc, inlier_ratio)
    except Exception:
        # ECC failed; return the ORB estimate.
        return matrix, inlier_ratio


def _find_feature_matches(
    ref: np.ndarray,
    curr: np.ndarray,
    feature_type: str = "orb",
    max_features: int = 1000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Find corresponding feature points between two frames.

    Returns (src_pts, dst_pts, match_score) where src_pts are points in ``curr``
    and dst_pts are corresponding points in ``ref``.  Returns (None, None, 0.0)
    if matching fails.
    """
    try:
        if feature_type == "sift":
            detector = cv2.SIFT_create(max_features)
        else:
            detector = cv2.ORB_create(max_features)

        kp1, des1 = detector.detectAndCompute(ref, None)
        kp2, des2 = detector.detectAndCompute(curr, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING if feature_type == "orb" else cv2.NORM_L2)
        matches = bf.knnMatch(des2, des1, k=2)

        # Apply Lowe's ratio test to filter good matches.
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) < 8:
            return None, None, 0.0

        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Compute match score as average distance ratio.
        score = 1.0 - (sum(m.distance for m in good) / len(good)) / 256.0
        score = max(0.0, min(1.0, score))

        return src_pts, dst_pts, score
    except Exception:
        return None, None, 0.0


def _estimate_homography(
    ref: np.ndarray,
    curr: np.ndarray,
    feature_type: str = "orb",
    max_features: int = 1000,
    use_ecc: bool = True,
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate homography or affine transform between two frames.

    Uses feature matching to find corresponding points, then RANSAC to estimate
    the transform.  Optionally refines with ECC for sub-pixel accuracy.

    Returns (matrix, confidence) where matrix is a 3x3 homography or 2x3 affine
    matrix mapping points from ``curr`` to ``ref``.
    """
    src_pts, dst_pts, match_score = _find_feature_matches(
        ref, curr, feature_type, max_features
    )

    if src_pts is None or dst_pts is None:
        return None, 0.0

    # Estimate homography with RANSAC.
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0.0

    # Compute inlier ratio as confidence.
    inlier_ratio = float(np.sum(mask)) / len(mask) if mask is not None else 0.0
    confidence = min(match_score, inlier_ratio)

    # Optionally refine with ECC for sub-pixel accuracy.
    if use_ecc:
        try:
            # Convert homography to affine for ECC refinement.
            H_affine = H[:2, :].astype(np.float32)
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.0001)
            cc, H_refined = cv2.findTransformECC(
                ref_gray,
                curr_gray,
                H_affine,
                cv2.MOTION_HOMOGRAPHY,
                criteria,
                None,
                3,
            )
            # Convert back to 3x3 homography.
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = H_refined
            confidence = max(confidence, cc)
        except Exception:
            pass

    return H, confidence


def _homography_to_affine(H: np.ndarray) -> np.ndarray:
    """Convert 3x3 homography to 2x3 affine matrix for warpAffine."""
    return H[:2, :].astype(np.float32)


def _resize_for_analysis(frame: np.ndarray, max_dim: int = 320) -> np.ndarray:
    """Resize a frame to a small size suitable for fast motion analysis."""
    h, w = frame.shape[:2]
    if max(w, h) <= max_dim:
        return frame
    scale = max_dim / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _translation_matrix(dx: float, dy: float) -> np.ndarray:
    """Return a 2x3 pure translation matrix."""
    return np.float32([[1, 0, dx], [0, 1, dy]])


def _estimate_transform_ecc(
    ref: np.ndarray,
    curr: np.ndarray,
    motion_type: int = cv2.MOTION_TRANSLATION,
    initial_matrix: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Refine geometric transform using ECC.

    Supports MOTION_TRANSLATION or MOTION_EUCLIDEAN.  The optional
    initial_matrix seeds the optimizer so the overlap region is already
    roughly aligned.
    """
    try:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        if motion_type == cv2.MOTION_TRANSLATION:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            if initial_matrix is not None:
                warp_matrix[:, :] = initial_matrix
        elif motion_type == cv2.MOTION_EUCLIDEAN:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            if initial_matrix is not None:
                warp_matrix[:, :] = initial_matrix
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.0001)
        cc, warp_matrix = cv2.findTransformECC(
            ref_gray,
            curr_gray,
            warp_matrix,
            motion_type,
            criteria,
            None,
            3,
        )
        return warp_matrix.astype(np.float32), cc
    except Exception:
        if initial_matrix is not None:
            return initial_matrix.astype(np.float32), 0.0
        return None, 0.0


def _estimate_transform_scaled(
    ref: np.ndarray,
    curr: np.ndarray,
    ref_scale: Tuple[float, float],
    curr_scale: Tuple[float, float],
    use_ecc: bool = False,
    motion_type: int = cv2.MOTION_TRANSLATION,
) -> Tuple[Optional[np.ndarray], float]:
    """Estimate transform on small images and scale the result back.

    Returns a 2x3 affine matrix that maps points from ``curr`` coordinates to
    ``ref`` coordinates, scaled back to the original frame size.
    """
    sx, sy = ref_scale
    if use_ecc:
        # Seed ECC with phase correlation for robustness and speed.
        initial_shift, pc_response = _estimate_translation(ref, curr)
        initial_matrix = None
        if initial_shift is not None:
            initial_matrix = _translation_matrix(initial_shift[0], initial_shift[1])
        matrix, response = _estimate_transform_ecc(
            ref, curr, motion_type=motion_type, initial_matrix=initial_matrix
        )
        response = max(response, pc_response)
    else:
        shift, response = _estimate_translation(ref, curr)
        if shift is None or response < 0.3:
            # Phase correlation is unstable on repetitive textures; fall back
            # to ORB feature matching which handles sub-pixel shifts better.
            matrix, response = _estimate_transform_orb(ref, curr)
            if matrix is None:
                return None, response
        else:
            matrix = _translation_matrix(shift[0], shift[1])

    if matrix is None:
        return None, response

    # Scale translation part back to original resolution.
    matrix = matrix.copy()
    matrix[0, 2] /= sx
    matrix[1, 2] /= sy
    return matrix, response


def analyze_motion(
    frames: List[np.ndarray],
    direction: StitchDirection = StitchDirection.AUTO,
    analysis_max_dim: Optional[int] = None,
) -> MotionInfo:
    """Analyze dominant motion direction and estimated output size."""
    if not frames:
        return MotionInfo("无", 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0)

    h, w = frames[0].shape[:2]

    # Use thumbnails for motion analysis to keep this step fast, but adapt the
    # thumbnail size to the input resolution so high-res videos keep accuracy.
    if analysis_max_dim is None:
        analysis_max_dim = _analysis_max_dim(w, h)
    small_frames = [_resize_for_analysis(f, analysis_max_dim) for f in frames]
    small_h, small_w = small_frames[0].shape[:2]
    scale_x = small_w / w
    scale_y = small_h / h

    shifts = []

    for i in range(1, len(small_frames)):
        matrix, response = _estimate_transform_scaled(
            small_frames[i - 1],
            small_frames[i],
            (scale_x, scale_y),
            (scale_x, scale_y),
            use_ecc=False,
        )
        if matrix is not None and response >= 0.1:
            shifts.append((matrix[0, 2], matrix[1, 2]))

    if not shifts:
        return MotionInfo("无", 0.0, 0.0, 0.0, w, h, 0.0, 0.0)

    dx_values = [s[0] for s in shifts]
    dy_values = [s[1] for s in shifts]
    total_dx = sum(dx_values)
    total_dy = sum(dy_values)
    avg_dx = total_dx / len(shifts)
    avg_dy = total_dy / len(shifts)
    avg_shift = sum(np.hypot(*s) for s in shifts) / len(shifts)

    if direction == StitchDirection.HORIZONTAL:
        detected_dir = "水平"
        total_dy = 0.0
    elif direction == StitchDirection.VERTICAL:
        detected_dir = "垂直"
        total_dx = 0.0
    else:
        if abs(total_dx) >= abs(total_dy):
            detected_dir = "水平"
            total_dy = 0.0
        else:
            detected_dir = "垂直"
            total_dx = 0.0

    est_width = int(w + abs(total_dx))
    est_height = int(h + abs(total_dy))

    return MotionInfo(
        direction=detected_dir,
        avg_dx=avg_dx,
        avg_dy=avg_dy,
        avg_shift=avg_shift,
        estimated_width=est_width,
        estimated_height=est_height,
        total_shift_x=total_dx,
        total_shift_y=total_dy,
    )


def _analysis_max_dim(width: int, height: int, high_quality: bool = False) -> int:
    """Choose an analysis size that balances speed and accuracy.

    Higher-resolution videos benefit from a larger analysis image so that
    sub-pixel motion estimates stay accurate after scaling back.  In high
    quality mode we use a larger analysis image for better alignment.

    For high-overlap line-scan footage, tiny shifts (a few pixels) must be
    resolved accurately, so we use a much larger analysis size than before.
    """
    if high_quality:
        return max(1280, min(1920, max(width, height) // 2))
    return max(960, min(1280, max(width, height) // 3))


def _matrix_to_params(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Decompose a 2x3 affine matrix into (dx, dy, angle_degrees).

    The angle is approximate and assumes small rotation.
    """
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return dx, dy, angle


def _params_to_matrix(dx: float, dy: float, angle: float) -> np.ndarray:
    """Build a 2x3 Euclidean affine matrix from (dx, dy, angle_degrees)."""
    theta = np.radians(angle)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.float32([[c, -s, dx], [s, c, dy]])


def _to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Convert 2x3 affine matrix to 3x3 homogeneous matrix."""
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = matrix
    return h


def _from_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 homogeneous matrix back to 2x3 affine matrix."""
    return matrix[:2, :].astype(np.float32)


def _reject_outliers(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Return a boolean mask of inliers using median absolute deviation."""
    if len(values) < 3:
        return np.ones(len(values), dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-6:
        return np.ones(len(values), dtype=bool)
    return np.abs(values - median) < threshold * mad


def _smooth_1d(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Smooth a 1-D array with a Savitzky-Golay filter.

    Falls back to a moving average if the window is too large or scipy is
    unavailable.
    """
    try:
        from scipy.signal import savgol_filter

        window = min(window, len(values))
        if window % 2 == 0:
            window -= 1
        if window < 5:
            return values.copy()
        return savgol_filter(values, window_length=window, polyorder=3)
    except Exception:
        # Fallback moving average.
        half = window // 2
        result = values.copy()
        for i in range(half, len(values) - half):
            result[i] = values[i - half : i + half + 1].mean()
        return result


def _smooth_transforms(
    rel_transforms: List[np.ndarray], window: int = 7
) -> List[np.ndarray]:
    """Smooth a sequence of relative transforms.

    Outlier relative motions are rejected and the remaining parameters are
    smoothed with a Savitzky-Golay filter.  This removes handheld jitter
    without distorting the global motion path.
    """
    if len(rel_transforms) < 5:
        return rel_transforms

    params = np.array([_matrix_to_params(m) for m in rel_transforms], dtype=np.float64)
    dxs, dys, angles = params[:, 0], params[:, 1], params[:, 2]

    # Reject outliers in each parameter independently.
    valid_dx = _reject_outliers(dxs)
    valid_dy = _reject_outliers(dys)
    valid_angle = _reject_outliers(angles)
    valid = valid_dx & valid_dy & valid_angle

    # Replace outliers with the median of valid neighbors.
    for arr in (dxs, dys, angles):
        median = np.median(arr[valid]) if valid.any() else np.median(arr)
        arr[~valid] = median

    sdx = _smooth_1d(dxs, window=window)
    sdy = _smooth_1d(dys, window=window)
    sang = _smooth_1d(angles, window=window)

    return [
        _params_to_matrix(float(sdx[i]), float(sdy[i]), float(sang[i]))
        for i in range(len(rel_transforms))
    ]


def _compute_cumulative_transforms(
    frames: List[np.ndarray],
    direction: StitchDirection,
    analysis_max_dim: Optional[int] = None,
    use_ecc: bool = False,
    motion_type: int = cv2.MOTION_TRANSLATION,
    smooth: bool = False,
) -> List[np.ndarray]:
    """Compute cumulative affine transforms between consecutive frames.

    Returns a list of 2x3 affine matrices, one per frame, mapping points from
    each frame's coordinate system back to the first frame's coordinate system.
    """
    if not frames:
        return [_translation_matrix(0.0, 0.0)]

    h, w = frames[0].shape[:2]
    if analysis_max_dim is None:
        analysis_max_dim = _analysis_max_dim(w, h, high_quality=use_ecc)

    small_frames = [_resize_for_analysis(f, analysis_max_dim) for f in frames]
    small_h, small_w = small_frames[0].shape[:2]
    scale_x = small_w / w
    scale_y = small_h / h

    rel_transforms = [_translation_matrix(0.0, 0.0)]

    for i in range(1, len(small_frames)):
        # Phase correlation is more robust than ORB on repetitive textures like
        # plant panels, and translation is sufficient for one-direction panning.
        shift, response = _estimate_translation(small_frames[i - 1], small_frames[i])
        if shift is None or response < 0.1:
            # Fallback: tiny motion in dominant direction.
            if direction == StitchDirection.VERTICAL:
                matrix = _translation_matrix(0.0, 1.0)
            else:
                matrix = _translation_matrix(1.0, 0.0)
        else:
            matrix = _translation_matrix(shift[0], shift[1])

        # Scale translation part back to original resolution.
        matrix = matrix.copy()
        matrix[0, 2] /= scale_x
        matrix[1, 2] /= scale_y

        rel_transforms.append(matrix)

    if smooth and len(rel_transforms) >= 5:
        rel_transforms = _smooth_transforms(
            rel_transforms, window=min(7, len(rel_transforms) // 4 * 2 + 1)
        )

    # Chain relative transforms into cumulative transforms.
    cumulative = [_translation_matrix(0.0, 0.0)]
    cumulative_3x3 = np.eye(3, dtype=np.float64)
    for rel in rel_transforms[1:]:
        cumulative_3x3 = cumulative_3x3 @ _to_homogeneous(rel)
        cumulative.append(_from_homogeneous(cumulative_3x3))

    return cumulative


def _gray_world_gains(image: np.ndarray) -> np.ndarray:
    """Compute mild gray-world white balance gains for an image.

    Returns B/G/R gains that map the average color toward neutral gray.
    Gains are clamped to a narrow range to avoid over-correction.
    """
    means = np.array(
        [image[:, :, c].mean() for c in range(3)], dtype=np.float32
    )
    gray = means.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        gains = np.where(means > 1.0, gray / means, 1.0)
    return np.clip(gains, 0.9, 1.1)


def _gain_compensate(
    frames: List[np.ndarray],
    transforms: Optional[List[np.ndarray]] = None,
    high_quality: bool = False,
) -> List[np.ndarray]:
    """Apply color/exposure compensation to reduce frame-to-frame differences.

    In high quality mode:
      1. A mild gray-world white balance is applied using the first frame as a
         reference, so slow white-balance drifts are corrected without
         introducing per-frame color instability.
      2. Pairwise luminance matching in the overlap region adjusts brightness
         only, leaving hue/saturation untouched.  This avoids the purple/green
         casts caused by per-channel RGB gain compensation.

    When no transforms are provided or high_quality is False, it falls back to
    the legacy whole-frame brightness scaling.
    """
    if len(frames) <= 1:
        return frames

    if not high_quality or transforms is None or len(transforms) != len(frames):
        # Legacy whole-frame brightness compensation.
        reference = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        ref_mean = reference.mean()
        compensated = [frames[0]]
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean = gray.mean()
            if mean > 0:
                scale = ref_mean / mean
                scale = max(0.5, min(2.0, scale))
                adjusted = np.clip(frame * scale, 0, 255).astype(np.uint8)
                compensated.append(adjusted)
            else:
                compensated.append(frame)
        return compensated

    h, w = frames[0].shape[:2]

    # 1. Mild gray-world white balance relative to the first frame.
    ref_gains = _gray_world_gains(frames[0])
    corrected = [frames[0]]
    for frame in frames[1:]:
        gains = _gray_world_gains(frame) / ref_gains
        gains = np.clip(gains, 0.9, 1.1)
        corrected.append(np.clip(frame.astype(np.float32) * gains, 0, 255).astype(np.uint8))

    # 2. Pairwise luminance matching in overlap regions.
    result = [corrected[0]]
    for i in range(1, len(corrected)):
        curr = corrected[i].astype(np.float32)
        prev = result[i - 1]

        tx_prev, ty_prev, _ = _matrix_to_params(transforms[i - 1])
        tx_curr, ty_curr, _ = _matrix_to_params(transforms[i])
        dx = tx_curr - tx_prev
        dy = ty_curr - ty_prev

        overlap_h = max(0.0, h - abs(dy))
        overlap_w = max(0.0, w - abs(dx))
        scale = 1.0
        if overlap_h > 20 and overlap_w > 20:
            if abs(dx) >= abs(dy):
                strip_w = int(min(w, max(30, overlap_w)))
                if dx > 0:
                    curr_roi = curr[:, :strip_w]
                    prev_roi = prev[:, -strip_w:]
                else:
                    curr_roi = curr[:, -strip_w:]
                    prev_roi = prev[:, :strip_w]
            else:
                strip_h = int(min(h, max(30, overlap_h)))
                if dy > 0:
                    curr_roi = curr[:strip_h, :]
                    prev_roi = prev[-strip_h:, :]
                else:
                    curr_roi = curr[-strip_h:, :]
                    prev_roi = prev[:strip_h, :]

            curr_lum = float(cv2.cvtColor(curr_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).mean())
            prev_lum = float(cv2.cvtColor(prev_roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).mean())
            if curr_lum > 1.0 and prev_lum > 1.0:
                scale = max(0.9, min(1.1, prev_lum / curr_lum))

        adjusted = np.clip(curr * scale, 0, 255).astype(np.uint8)
        result.append(adjusted)

    return result


def _build_distance_weight(mask: np.ndarray, feather: int = 15) -> np.ndarray:
    """Create a smooth feather weight map from a binary mask.

    Pixels near the center of the frame get weight 1, pixels near the edge
    taper to 0 over `feather` pixels.  This eliminates hard edges in the
    overlap area, which is the main cause of visible seams and color jumps.
    """
    # Distance transform: distance to the nearest zero pixel.
    dist = cv2.distanceTransform(
        (mask > 0.5).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    if feather <= 0:
        return (dist > 0).astype(np.float32)
    weight = np.clip(dist / feather, 0.0, 1.0).astype(np.float32)
    return weight


def _gaussian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build a Gaussian pyramid as float32 images."""
    image = image.astype(np.float32)
    gp = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        gp.append(image)
    return gp


def _reconstruct_laplacian(lp: List[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from its Laplacian pyramid."""
    image = lp[-1].copy()
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size) + lp[i]
    return image


def _multiband_blend(
    warped_frames: List[np.ndarray],
    weight_maps: List[np.ndarray],
    levels: int = 4,
) -> np.ndarray:
    """Blend multiple aligned frames using multi-band blending.

    Multi-band blending decomposes images into Laplacian pyramids and blends
    each frequency band separately.  This hides low-frequency color/brightness
    seams while preserving high-frequency detail, which is much more forgiving
    than a single weighted average when exposures or white balance differ.

    This implementation accumulates each pyramid level on the fly so memory
    usage does not grow with the number of frames.
    """
    if not warped_frames:
        return None

    # Determine pyramid level sizes from the first frame.
    first_gp = _gaussian_pyramid(warped_frames[0], levels)
    accumulator_num = [np.zeros_like(g) for g in first_gp]
    accumulator_den = [
        np.zeros(g.shape[:2], dtype=np.float32) for g in first_gp
    ]

    for frame, weight in zip(warped_frames, weight_maps):
        # Build Gaussian pyramid for this frame.
        frame_gp = _gaussian_pyramid(frame, levels)
        # Build Gaussian pyramid for weight.
        weight_gp = _gaussian_pyramid(weight, levels)

        # Accumulate coarsest level (Laplacian == Gaussian there).
        l = levels - 1
        w = weight_gp[l]
        if len(w.shape) == 2:
            w = w[:, :, np.newaxis]
        accumulator_num[l] += frame_gp[l] * w
        accumulator_den[l] += w.squeeze()

        # Accumulate remaining levels as Laplacians.
        for l in range(levels - 2, -1, -1):
            size = (frame_gp[l].shape[1], frame_gp[l].shape[0])
            up = cv2.pyrUp(frame_gp[l + 1], dstsize=size)
            lap = frame_gp[l] - up
            w = weight_gp[l]
            if len(w.shape) == 2:
                w = w[:, :, np.newaxis]
            accumulator_num[l] += lap * w
            accumulator_den[l] += w.squeeze()

    # Normalize each level.
    blended_levels = []
    for num, den in zip(accumulator_num, accumulator_den):
        den = np.maximum(den, 1e-6)
        if len(num.shape) == 3:
            blended_levels.append(num / den[:, :, np.newaxis])
        else:
            blended_levels.append(num / den)

    result = _reconstruct_laplacian(blended_levels)
    return np.clip(result, 0, 255).astype(np.uint8)


def _mild_unsharp_mask(image: np.ndarray, amount: float = 0.4) -> np.ndarray:
    """Apply a very mild unsharp mask to restore slight softening from warps.

    The kernel is small (radius 1) and the amount is low to avoid amplifying
    noise or creating halos.
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0.5)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _compose_sequential(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
    blend: bool = True,
    fast_blend: bool = False,
    high_quality: bool = False,
    progress_callback=None,
) -> Optional[np.ndarray]:
    """Compose frames onto a canvas using accumulated affine transforms.

    Uses sub-pixel affine warping so frames are placed at their exact offsets.
    In high quality mode, multi-band blending is used to hide low-frequency
    color/brightness seams while preserving detail.
    """
    if not frames or not transforms:
        return None

    h, w = frames[0].shape[:2]
    translations = [_matrix_to_params(m) for m in transforms]
    min_x = min(0.0, min(t[0] for t in translations))
    max_x = max(0.0, max(t[0] for t in translations))
    min_y = min(0.0, min(t[1] for t in translations))
    max_y = max(0.0, max(t[1] for t in translations))

    # Allow a one-pixel margin for sub-pixel placement.
    canvas_w = int(np.ceil(w + max_x - min_x))
    canvas_h = int(np.ceil(h + max_y - min_y))

    if canvas_w <= 0 or canvas_h <= 0 or canvas_w > 30000 or canvas_h > 30000:
        return None

    offset_x = -min_x
    offset_y = -min_y

    total = len(frames)
    src_mask = np.ones((h, w), dtype=np.float32)

    if blend and not fast_blend and high_quality:
        # High quality path: multi-band blending with distance-feather weights.
        warped_frames = []
        weight_maps = []

        for i, (frame, M) in enumerate(zip(frames, transforms)):
            M_canvas = M.copy()
            M_canvas[0, 2] += offset_x
            M_canvas[1, 2] += offset_y

            warped = cv2.warpAffine(
                frame,
                M_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            mask = cv2.warpAffine(
                src_mask,
                M_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            # Adaptive feather: at least 6% of the smaller dimension or 10% of
            # the estimated overlap in the dominant direction.
            tx, ty, _ = _matrix_to_params(M)
            if i > 0:
                tx_prev, ty_prev, _ = _matrix_to_params(transforms[i - 1])
                overlap_w = max(0.0, w - abs(tx - tx_prev))
                overlap_h = max(0.0, h - abs(ty - ty_prev))
            else:
                overlap_w, overlap_h = float(w), float(h)
            dominant_overlap = max(overlap_w, overlap_h)
            feather = max(12, int(min(h, w) * 0.06), int(dominant_overlap * 0.10))
            wmap = _build_distance_weight(mask, feather=feather)

            warped_frames.append(warped)
            weight_maps.append(wmap)

            if progress_callback and (i % 5 == 0 or i == total - 1):
                progress_callback(i + 1, total)

        canvas = _multiband_blend(warped_frames, weight_maps, levels=4)
        if canvas is not None and high_quality:
            canvas = _mild_unsharp_mask(canvas, amount=0.25)

    elif blend and not fast_blend:
        # Standard path: simple weighted average with binary masks.
        canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for i, (frame, M) in enumerate(zip(frames, transforms)):
            M_canvas = M.copy()
            M_canvas[0, 2] += offset_x
            M_canvas[1, 2] += offset_y

            warped = cv2.warpAffine(
                frame.astype(np.float32),
                M_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            mask = cv2.warpAffine(
                src_mask,
                M_canvas,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            canvas_sum += warped
            weight += mask

            if progress_callback and (i % 5 == 0 or i == total - 1):
                progress_callback(i + 1, total)

        weight = np.maximum(weight, 1.0)
        canvas = (canvas_sum / weight[:, :, np.newaxis]).astype(np.uint8)

    else:
        # Fast path: simple overwrite.
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i, (frame, M) in enumerate(zip(frames, transforms)):
            M_canvas = M.copy()
            M_canvas[0, 2] += offset_x
            M_canvas[1, 2] += offset_y
            cv2.warpAffine(
                frame,
                M_canvas,
                (canvas_w, canvas_h),
                dst=canvas,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            if progress_callback and (i % 5 == 0 or i == total - 1):
                progress_callback(i + 1, total)

    return canvas


def _panorama_stitch(
    frames: List[np.ndarray],
    feature_type: str = "orb",
    blend_width: int = 50,
    blend_type: str = "linear",
) -> Optional[np.ndarray]:
    """Stitch frames using feature alignment and linear gradient blending.

    Each frame is aligned to the previous one using feature matching (SIFT/ORB)
    and homography/affine transform.  Overlapping regions are blended with a
    gradient mask (linear or cosine) so transitions are smooth and invisible.
    """
    if not frames:
        return None

    n = len(frames)
    if n == 1:
        return frames[0]

    # Step 1: Estimate homography between consecutive frames.
    transforms = [np.eye(3, dtype=np.float64)]
    for i in range(1, n):
        H, confidence = _estimate_homography(
            frames[i - 1], frames[i], feature_type=feature_type
        )
        if H is None or confidence < 0.3:
            # Fallback: pure translation via phase correlation.
            shift, response = _estimate_translation(frames[i - 1], frames[i])
            if shift is None:
                H = np.eye(3, dtype=np.float64)
            else:
                H = np.eye(3, dtype=np.float64)
                H[0, 2] = shift[0]
                H[1, 2] = shift[1]
        transforms.append(H)

    # Step 2: Chain transforms to get cumulative positions.
    cumulative = [np.eye(3, dtype=np.float64)]
    for H in transforms[1:]:
        cumulative.append(cumulative[-1] @ H)

    # Step 3: Compute panorama bounds from all transformed corners.
    h, w = frames[0].shape[:2]
    corners = np.array([
        [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]
    ], dtype=np.float64).T

    min_x, min_y = 0.0, 0.0
    max_x, max_y = float(w), float(h)
    for H in cumulative:
        tc = (H @ corners).T
        min_x = min(min_x, float(tc[:, 0].min()))
        max_x = max(max_x, float(tc[:, 0].max()))
        min_y = min(min_y, float(tc[:, 1].min()))
        max_y = max(max_y, float(tc[:, 1].max()))

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))
    if canvas_w <= 0 or canvas_h <= 0 or canvas_w > 30000 or canvas_h > 30000:
        return None

    offset_x = -min_x
    offset_y = -min_y

    # Step 4: Warp and blend each frame with gradient mask.
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    src_mask = np.ones((h, w), dtype=np.float32)

    for i, (frame, H) in enumerate(zip(frames, cumulative)):
        H_canvas = H.copy()
        H_canvas[0, 2] += offset_x
        H_canvas[1, 2] += offset_y

        # Warp frame and mask to panorama.
        warped = cv2.warpPerspective(
            frame,
            H_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        mask = cv2.warpPerspective(
            src_mask,
            H_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        # Distance-based gradient mask for smooth blending.
        dist = cv2.distanceTransform(
            (mask > 0.5).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        if blend_type == "cosine":
            # Cosine gradient: smoother transition.
            wmap = np.clip(dist / max(blend_width, 1), 0.0, 1.0).astype(np.float32)
            wmap = (1 - np.cos(wmap * np.pi)) / 2.0
        else:
            # Linear gradient: simple and effective.
            wmap = np.clip(dist / max(blend_width, 1), 0.0, 1.0).astype(np.float32)

        canvas_sum += warped * wmap[:, :, np.newaxis]
        weight_sum += wmap

    weight_sum = np.maximum(weight_sum, 1e-6)
    canvas = (canvas_sum / weight_sum[:, :, np.newaxis]).astype(np.uint8)

    return canvas


def _final_smooth(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a slight bilateral filter to smooth texture while preserving edges."""
    if sigma <= 0:
        return image
    try:
        d = max(5, int(sigma * 3))
        return cv2.bilateralFilter(image, d, sigma * 50, sigma * 50)
    except Exception:
        return image


def _seamless_stitch(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
    seam_width: int = 30,
    seam_blur: int = 5,
) -> Optional[np.ndarray]:
    """Stitch frames using affine transforms with seamless blending.

    Each frame is warped to the panorama using its affine transform.  Overlapping
    regions are blended with a gradient mask, and a slight Gaussian blur is
    applied at the seams to hide transitions.  This avoids both ghosting and
    hard edges.
    """
    if not frames or not transforms:
        return None

    h, w = frames[0].shape[:2]
    n = len(frames)

    # Compute panorama bounds from all transformed corners.
    corners = np.array([
        [0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]
    ], dtype=np.float32).T

    min_x, min_y = 0.0, 0.0
    max_x, max_y = float(w), float(h)
    for M in transforms:
        M3 = _to_homogeneous(M)
        tc = (M3 @ corners).T
        min_x = min(min_x, float(tc[:, 0].min()))
        max_x = max(max_x, float(tc[:, 0].max()))
        min_y = min(min_y, float(tc[:, 1].min()))
        max_y = max(max_y, float(tc[:, 1].max()))

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))
    if canvas_w <= 0 or canvas_h <= 0 or canvas_w > 30000 or canvas_h > 30000:
        return None

    offset_x = -min_x
    offset_y = -min_y

    # Accumulate frames with distance-based weights.
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    src_mask = np.ones((h, w), dtype=np.float32)

    for i, (frame, M) in enumerate(zip(frames, transforms)):
        M_canvas = M.copy()
        M_canvas[0, 2] += offset_x
        M_canvas[1, 2] += offset_y

        warped = cv2.warpAffine(
            frame,
            M_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        mask = cv2.warpAffine(
            src_mask,
            M_canvas,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

        # Distance-based weight: pixels near the frame edge get lower weight.
        # This creates a natural gradient at the seams.
        dist = cv2.distanceTransform(
            (mask > 0.5).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        # Normalize to 0-1 over the seam width.
        wmap = np.clip(dist / max(seam_width, 1), 0.0, 1.0).astype(np.float32)

        canvas_sum += warped * wmap[:, :, np.newaxis]
        weight_sum += wmap

    weight_sum = np.maximum(weight_sum, 1e-6)
    canvas = (canvas_sum / weight_sum[:, :, np.newaxis]).astype(np.uint8)

    # Apply a slight Gaussian blur at the seams to hide transitions.
    # Create a mask of pixels where weight_sum is between 0 and 1 (the seams).
    seam_mask = ((weight_sum > 0) & (weight_sum < 1)).astype(np.uint8)
    if np.any(seam_mask):
        blurred = cv2.GaussianBlur(canvas, (2 * seam_blur + 1, 2 * seam_blur + 1), 0)
        # Blend blurred and original only at the seams.
        canvas = np.where(
            seam_mask[:, :, np.newaxis] > 0,
            blurred,
            canvas,
        ).astype(np.uint8)

    return canvas


def _final_smooth(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a slight bilateral filter to smooth texture while preserving edges."""
    if sigma <= 0:
        return image
    try:
        # Bilateral filter preserves edges while smoothing texture.
        d = max(5, int(sigma * 3))
        return cv2.bilateralFilter(image, d, sigma * 50, sigma * 50)
    except Exception:
        return image


def _strip_stitch(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
    overlap_ratio: float = 0.0,
    safety_margin: int = 10,
    edge_crop_ratio: float = 0.1,
) -> Optional[np.ndarray]:
    """Stitch frames using a pure line-scan approach with hard stitching.

    Each frame contributes a central strip.  The strip is placed directly
    without any blending or feathering, so there is no ghosting or blur.
    The edges of each frame are cropped by ``edge_crop_ratio`` to remove
    potential distortion or blur at the frame edges.

    The canvas includes a safety margin so the main content is never cropped.
    """
    if not frames or not transforms:
        return None

    # Crop 10% from each frame's edges to remove distortion/blur.
    processed_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        crop_x = int(w * edge_crop_ratio)
        crop_y = int(h * edge_crop_ratio)
        if crop_x > 0 and crop_y > 0:
            frame = frame[crop_y : h - crop_y, crop_x : w - crop_x]
        processed_frames.append(frame)

    frames = processed_frames

    h, w = frames[0].shape[:2]
    translations = [_matrix_to_params(m) for m in transforms]
    n = len(frames)

    if n == 1:
        return frames[0]

    total_dx = abs(translations[-1][0] - translations[0][0])
    total_dy = abs(translations[-1][1] - translations[0][1])
    is_horizontal = total_dx >= total_dy

    # Extract per-frame positions along the dominant axis.
    positions = [t[0] if is_horizontal else t[1] for t in translations]
    min_pos = min(positions)
    max_pos = max(positions)
    total_span = max_pos - min_pos
    avg_step = total_span / (n - 1) if n > 1 else 0.0

    if avg_step < 0.5:
        return None

    # Use the maximum step with only 10% overlap so strips cover gaps but
    # almost never blend with neighbors.  This keeps every frame clearly
    # visible (no ghosting) while avoiding black lines from uneven speed.
    effective_overlap = max(overlap_ratio, 0.1)
    max_step = max(
        abs(positions[i + 1] - positions[i]) for i in range(n - 1)
    ) if n > 1 else avg_step

    if is_horizontal:
        strip_width = max(1, min(int(max_step * (1.0 + effective_overlap)), w))
        canvas_width = int(total_span + strip_width + 2 * safety_margin)
        canvas_height = h + 2 * safety_margin
        if canvas_width <= 0 or canvas_height <= 0 or canvas_width > 30000 or canvas_height > 30000:
            return None

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        for i, (frame, M) in enumerate(zip(frames, transforms)):
            tx, _, _ = _matrix_to_params(M)
            pano_x = int(tx - min_pos) + safety_margin
            pano_y = safety_margin

            center = w // 2
            x1 = max(0, center - strip_width // 2)
            x2 = min(w, x1 + strip_width)
            actual_w = x2 - x1
            if actual_w <= 0:
                continue

            strip = frame[:, x1:x2]

            # Hard stitching: place the strip directly without any blending.
            x_end = min(canvas_width, pano_x + actual_w)
            placed_w = x_end - pano_x
            canvas[pano_y : pano_y + h, pano_x:x_end] = strip[:, :placed_w]

        return canvas

    # Vertical movement.
    strip_height = max(1, min(int(max_step * (1.0 + effective_overlap)), h))
    canvas_height = int(total_span + strip_height + 2 * safety_margin)
    canvas_width = w + 2 * safety_margin
    if canvas_width <= 0 or canvas_height <= 0 or canvas_width > 30000 or canvas_height > 30000:
        return None

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for i, (frame, M) in enumerate(zip(frames, transforms)):
        _, ty, _ = _matrix_to_params(M)
        pano_y = int(ty - min_pos) + safety_margin
        pano_x = safety_margin

        center = h // 2
        y1 = max(0, center - strip_height // 2)
        y2 = min(h, y1 + strip_height)
        actual_h = y2 - y1
        if actual_h <= 0:
            continue

        strip = frame[y1:y2, :]

        # Hard stitching: place the strip directly without any blending.
        y_end = min(canvas_height, pano_y + actual_h)
        placed_h = y_end - pano_y
        canvas[pano_y:y_end, pano_x : pano_x + w] = strip[:placed_h, :]

    return canvas


def _auto_stitch(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """Try OpenCV Stitcher."""
    try:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(1.0)
        status, result = stitcher.stitch(frames)
        if status == cv2.Stitcher_OK:
            return result
    except Exception:
        pass
    return None


def _remove_background(
    frame: np.ndarray,
    bg_threshold: float = 0.1,
    bg_margin: int = 20,
) -> np.ndarray:
    """Remove pure-color borders and device structures from a frame.

    Detects the water-grass subject using color thresholding (green hue) and
    edge detection, then crops to the subject's bounding box with a small
    margin.  This eliminates white panels, gray backgrounds, and other
    non-subject content before stitching.
    """
    h, w = frame.shape[:2]

    # Convert to HSV and threshold for green (water grass).
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Green hue range: restrictive threshold to avoid including white/gray
    # borders that might have a slight green tint.
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask with morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Find contours and take the largest one (the water grass).
    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return frame

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (h * w * bg_threshold):
        # Subject is too small; don't crop.
        return frame

    x, y, bw, bh = cv2.boundingRect(largest)

    # Add margin around the subject.
    x1 = max(0, x - bg_margin)
    y1 = max(0, y - bg_margin)
    x2 = min(w, x + bw + bg_margin)
    y2 = min(h, y + bh + bg_margin)

    return frame[y1:y2, x1:x2]


def _crop_black_borders(
    image: np.ndarray,
    threshold: int = 5,
    safety_margin: int = 5,
) -> np.ndarray:
    """Crop black borders from composed image with safety margin.

    Only crops pure black regions (gray < threshold) and keeps a safety
    margin so the main content is never cropped.
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)

    # Add safety margin so we don't crop into the main content.
    x = max(0, x - safety_margin)
    y = max(0, y - safety_margin)
    w = min(image.shape[1] - x, w + 2 * safety_margin)
    h = min(image.shape[0] - y, h + 2 * safety_margin)

    return image[y : y + h, x : x + w]


def _is_likely_single_frame(
    result: np.ndarray,
    original: np.ndarray,
    motion: MotionInfo,
) -> bool:
    """Check if the stitched result is suspiciously similar to a single frame."""
    rh, rw = result.shape[:2]
    oh, ow = original.shape[:2]

    if abs(rw - ow) < ow * 0.1 and abs(rh - oh) < oh * 0.1:
        return True

    if abs(motion.total_shift_x) < ow * 0.05 and abs(motion.total_shift_y) < oh * 0.05:
        return True

    return False


def stitch_frames(
    frames: List[np.ndarray],
    mode: StitchMode = StitchMode.SEQUENTIAL,
    direction: StitchDirection = StitchDirection.AUTO,
    overlap_ratio: float = 0.0,
    preview_scale: float = 1.0,
    output_scale: float = 1.0,
    blend: bool = True,
    fast_blend: bool = False,
    exposure_compensate: bool = True,
    high_quality: bool = False,
    progress_callback=None,
) -> StitchResult:
    """Stitch a list of frames into a single image."""
    if not frames:
        return StitchResult(False, None, "没有可用的帧进行拼接")

    if len(frames) == 1:
        return StitchResult(
            True,
            _resize_for_save(frames[0].copy(), output_scale, high_quality=high_quality),
            "只有一帧，直接返回",
            is_likely_single_frame=True,
        )

    motion = analyze_motion(frames, direction)
    # Do not reject videos with small motion; even nearly static frames can be
    # concatenated for documentation purposes.
    if motion.avg_shift < 0.0:
        return StitchResult(
            False,
            None,
            "检测到帧间几乎没有移动，请检查视频是否真的在移动，或减小抽帧间隔",
        )

    # For vertical/horizontal videos, force sequential translation-only to avoid black holes
    actual_mode = mode
    actual_direction = direction
    if direction == StitchDirection.AUTO:
        if motion.direction == "垂直":
            actual_direction = StitchDirection.VERTICAL
            # Auto mode should not use OpenCV Stitcher for vertical translation
            if mode == StitchMode.AUTO:
                actual_mode = StitchMode.SEQUENTIAL
        elif motion.direction == "水平":
            actual_direction = StitchDirection.HORIZONTAL

    # Limit canvas size for preview to avoid memory/time explosion.
    effective_preview_scale = preview_scale
    max_canvas = 8000 if preview_scale < 1.0 else 16000
    if motion.estimated_width * preview_scale > max_canvas or motion.estimated_height * preview_scale > max_canvas:
        scale_x = max_canvas / max(motion.estimated_width, 1)
        scale_y = max_canvas / max(motion.estimated_height, 1)
        effective_preview_scale = min(preview_scale, scale_x, scale_y)

    working_frames = [
        _resize_image(f, effective_preview_scale, high_quality=high_quality)
        for f in frames
    ]

    # Use pure translation even in high quality mode.  Real handheld videos
    # usually translate far more than they rotate, and estimating rotation on
    # small analysis frames introduces cumulative alignment errors that make
    # seams visible and edges jagged.
    motion_type = cv2.MOTION_TRANSLATION
    transforms = _compute_cumulative_transforms(
        working_frames,
        actual_direction,
        use_ecc=high_quality,
        motion_type=motion_type,
        smooth=high_quality,
    )

    if exposure_compensate:
        working_frames = _gain_compensate(
            working_frames,
            transforms if high_quality else None,
            high_quality=high_quality,
        )

    def _progress(step: int, total: int):
        if progress_callback:
            progress_callback(step, total)

    result_image = None
    method_used = ""

    if actual_mode == StitchMode.AUTO:
        result_image = _auto_stitch(working_frames)
        if result_image is not None:
            method_used = "OpenCV Stitcher"
        else:
            result_image = _compose_sequential(
                working_frames,
                transforms,
                blend=blend,
                fast_blend=fast_blend,
                high_quality=high_quality,
                progress_callback=_progress,
            )
            method_used = "顺序拼接（OpenCV Stitcher 失败后的回退，高质量）" if high_quality else "顺序拼接（OpenCV Stitcher 失败后的回退）"
    elif actual_mode == StitchMode.SEQUENTIAL:
        result_image = _compose_sequential(
            working_frames,
            transforms,
            blend=blend,
            fast_blend=fast_blend,
            high_quality=high_quality,
            progress_callback=_progress,
        )
        method_used = "顺序拼接（高质量）" if high_quality else "顺序拼接"
    elif actual_mode == StitchMode.STRIP:
        # Use direct pixel concatenation in a grid layout without any blending.
        # Each frame's edges are cropped by 10% to remove motion blur or distortion.
        # The grid dimensions are chosen to avoid overly long or wide output.
        direction_str = "horizontal" if actual_direction == StitchDirection.HORIZONTAL else "vertical"
        result_image = _grid_stitch(
            working_frames,
            direction=direction_str,
            crop_ratio=0.1,
            max_aspect=2.0,
        )
        method_used = "直接像素拼接（网格布局）"
    else:
        return StitchResult(False, None, f"未知的拼接模式: {mode}")

    if result_image is None:
        return StitchResult(False, None, "拼接失败，请尝试调整参数或换用其他模式")

    # Remove background after stitching so only the water grass subject is kept.
    result_image = _remove_background(result_image, bg_threshold=0.1, bg_margin=20)

    result_image = _crop_black_borders(result_image, threshold=5, safety_margin=5)
    if result_image.size == 0:
        return StitchResult(False, None, "拼接结果为空")

    # Resize to target resolution to ensure consistent output size.
    result_image = _resize_to_target(
        result_image,
        target_resolution=(1920, 1080),
        max_resolution=(3840, 2160),
    )

    # Apply output scaling
    result_image = _resize_for_save(
        result_image, output_scale, high_quality=high_quality
    )

    is_single = _is_likely_single_frame(result_image, working_frames[0], motion)
    msg = f"拼接成功，使用 {method_used}"
    if is_single:
        msg += "（注意：结果尺寸接近单帧，建议减小抽帧间隔）"

    return StitchResult(
        True,
        result_image,
        msg,
        method_used,
        is_likely_single_frame=is_single,
    )


def _grid_stitch(
    frames: List[np.ndarray],
    direction: str = "horizontal",
    crop_ratio: float = 0.1,
    max_aspect: float = 2.0,
) -> Optional[np.ndarray]:
    """Stitch frames using direct pixel concatenation in a grid layout.

    Each frame is cropped by ``crop_ratio`` from its edges to remove motion
    blur or distortion, then concatenated directly with ``np.concatenate``
    in a grid layout.  The grid dimensions are chosen so that the output
    aspect ratio does not exceed ``max_aspect``, avoiding overly long or
    wide images.  No alignment, blending, feathering, or gradient transitions
    are used.
    """
    if not frames:
        return None

    if len(frames) == 1:
        return frames[0]

    # Crop edges from each frame to remove motion blur or distortion.
    processed_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        crop_x = int(w * crop_ratio)
        crop_y = int(h * crop_ratio)
        if crop_x > 0 and crop_y > 0:
            frame = frame[crop_y : h - crop_y, crop_x : w - crop_x]
        processed_frames.append(frame)

    n = len(processed_frames)
    h, w = processed_frames[0].shape[:2]

    # Choose grid dimensions to keep the output aspect ratio within max_aspect.
    # For horizontal: try 1×N first, if too wide then 2×N.
    # For vertical: try N×1 first, if too tall then N×2.
    if direction == "horizontal":
        # Try 1 row.
        aspect = (w * n) / h
        if aspect <= max_aspect:
            return np.concatenate(processed_frames, axis=1)
        # Too wide; use 2 rows.
        cols = int(np.ceil(n / 2))
        rows = 2
    else:
        # Try 1 column.
        aspect = w / (h * n)
        if aspect <= max_aspect:
            return np.concatenate(processed_frames, axis=0)
        # Too tall; use 2 columns.
        cols = 2
        rows = int(np.ceil(n / 2))

    # Pad the grid with empty frames if needed.
    while len(processed_frames) < rows * cols:
        processed_frames.append(np.zeros_like(processed_frames[0]))

    # Build the grid.
    grid_rows = []
    for r in range(rows):
        row_frames = processed_frames[r * cols : (r + 1) * cols]
        grid_rows.append(np.concatenate(row_frames, axis=1))

    result = np.concatenate(grid_rows, axis=0)
    return result


def _resize_to_target(
    image: np.ndarray,
    target_resolution: Tuple[int, int] = (1920, 1080),
    max_resolution: Tuple[int, int] = (3840, 2160),
) -> np.ndarray:
    """Resize an image to the target resolution with aspect ratio preserved.

    The image is scaled proportionally so that it fits within the target
    resolution without being too long or too wide.  The output resolution
    is guaranteed to be at least the target resolution and at most the
    maximum resolution.
    """
    target_w, target_h = target_resolution
    max_w, max_h = max_resolution
    h, w = image.shape[:2]

    # Compute scale to fit within target resolution.
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # Ensure output is at least target resolution.
    if new_w < target_w or new_h < target_h:
        scale = max(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

    # Ensure output does not exceed max resolution.
    if new_w > max_w or new_h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
