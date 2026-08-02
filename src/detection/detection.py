"""Detection helpers for Rune arrows and classic-client player markers."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.common import utils

try:
    import tensorflow as tf
except ImportError:
    tf = None


Point = Tuple[float, float]
PlayerCandidate = Tuple[Point, float]


class TensorFlowUnavailableError(RuntimeError):
    """Raised when Rune detection is requested without TensorFlow installed."""


def tensorflow_available() -> bool:
    return tf is not None


def load_model():
    """Load the optional Rune model, returning None when TensorFlow is absent."""
    if tf is None:
        print(
            "\n[~] TensorFlow is not installed; Rune model support is disabled. "
            "Core capture and GUI features remain available."
        )
        return None

    model_dir = "assets/models/rune_model_rnn_filtered_cannied/saved_model"
    try:
        return tf.saved_model.load(model_dir)
    except Exception as exc:
        print(f"\n[!] Rune model could not be loaded: {exc}")
        print("[~] Continuing with Rune model support disabled")
        return None


def classic_player_candidates(minimap: np.ndarray) -> List[PlayerCandidate]:
    """Find the yellow self-marker used by the Traditional Chinese classic UI.

    In this client, yellow represents the local player while red represents other
    players. Red pixels are explicitly rejected. The search is bounded so noisy
    minimaps cannot stall the capture thread.
    """
    if minimap is None or minimap.size == 0 or minimap.ndim != 3:
        return []

    bgr = minimap[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Local player: bright yellow / yellow-orange. The two ranges tolerate display
    # scaling and slight color shifts while staying well away from red markers.
    mask_yellow = cv2.inRange(hsv, (20, 105, 145), (39, 255, 255))
    mask_yellow_orange = cv2.inRange(hsv, (14, 135, 170), (25, 255, 255))
    mask = cv2.bitwise_or(mask_yellow, mask_yellow_orange)

    # Other players are red. Remove both red hue bands before component analysis.
    mask_red_low = cv2.inRange(hsv, (0, 90, 110), (10, 255, 255))
    mask_red_high = cv2.inRange(hsv, (170, 90, 110), (179, 255, 255))
    red_mask = cv2.bitwise_or(mask_red_low, mask_red_high)
    mask[red_mask > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = minimap.shape[:2]
    candidates: List[PlayerCandidate] = []

    for index in range(1, count):
        x, y, component_w, component_h, area = stats[index]
        if not 2 <= area <= 110:
            continue
        if not 2 <= component_w <= 18 or not 2 <= component_h <= 18:
            continue

        aspect = component_w / max(component_h, 1)
        if not 0.35 <= aspect <= 2.8:
            continue

        cx, cy = centroids[index]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + component_w), min(height, y + component_h)
        component_hsv = hsv[y1:y2, x1:x2]
        component_mask = mask[y1:y2, x1:x2] > 0
        if component_hsv.size == 0 or not np.any(component_mask):
            continue

        hues = component_hsv[:, :, 0][component_mask]
        saturation = float(np.mean(component_hsv[:, :, 1][component_mask])) / 255.0
        value = float(np.mean(component_hsv[:, :, 2][component_mask])) / 255.0
        mean_hue = float(np.mean(hues))

        # Reject components whose average hue drifts toward orange/red.
        if not 14.0 <= mean_hue <= 39.0:
            continue

        fill = area / float(max(component_w * component_h, 1))
        size_score = 1.0 - min(1.0, abs(area - 14.0) / 55.0)
        hue_score = 1.0 - min(1.0, abs(mean_hue - 28.0) / 16.0)
        confidence = min(
            0.99,
            0.28
            + 0.22 * saturation
            + 0.20 * value
            + 0.12 * fill
            + 0.10 * size_score
            + 0.08 * hue_score,
        )
        point = (float(cx) / max(width, 1), float(cy) / max(height, 1))
        candidates.append((point, confidence))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:6]


def canny(image):
    image = cv2.Canny(image, 200, 300)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def filter_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (1, 100, 100), (75, 255, 255))
    color_mask = mask > 0
    arrows = np.zeros_like(image, np.uint8)
    arrows[color_mask] = image[color_mask]
    return arrows


def run_inference_for_single_image(model, image):
    if tf is None or model is None:
        raise TensorFlowUnavailableError("TensorFlow Rune detection is unavailable")

    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {
        key: value[0, :num_detections].numpy()
        for key, value in output_dict.items()
    }
    output_dict["num_detections"] = num_detections
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)
    return output_dict


def sort_by_confidence(model, image):
    output_dict = run_inference_for_single_image(model, image)
    zipped = list(
        zip(
            output_dict["detection_scores"],
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
        )
    )
    pruned = [item for item in zipped if item[0] > 0.5]
    pruned.sort(key=lambda item: item[0], reverse=True)
    return pruned[:4]


def get_boxes(model, image):
    output_dict = run_inference_for_single_image(model, image)
    zipped = list(
        zip(
            output_dict["detection_scores"],
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
        )
    )
    pruned = [item for item in zipped if item[0] > 0.5]
    pruned.sort(key=lambda item: item[0], reverse=True)
    return [item[1:] for item in pruned[:4]]


@utils.run_if_enabled
def merge_detection(model, image):
    if model is None:
        return []

    label_map = {1: "up", 2: "down", 3: "left", 4: "right"}
    converter = {"up": "right", "down": "left"}
    classes = []

    height, width, _ = image.shape
    cropped = image[120:height // 2, width // 4:3 * width // 4]
    filtered = filter_color(cropped)
    cannied = canny(filtered)

    height, width, _ = cannied.shape
    boxes = get_boxes(model, cannied)
    if len(boxes) == 4:
        y_mins = [box[0][0] for box in boxes]
        x_mins = [box[0][1] for box in boxes]
        y_maxes = [box[0][2] for box in boxes]
        x_maxes = [box[0][3] for box in boxes]
        left = int(round(min(x_mins) * width))
        right = int(round(max(x_maxes) * width))
        top = int(round(min(y_mins) * height))
        bottom = int(round(max(y_maxes) * height))
        rune_box = cannied[top:bottom, left:right]

        height, width, channels = rune_box.shape
        pad_height, pad_width = 384, 455
        preprocessed = np.full(
            (pad_height, pad_width, channels),
            (0, 0, 0),
            dtype=np.uint8,
        )
        x_offset = (pad_width - width) // 2
        y_offset = (pad_height - height) // 2

        if x_offset > 0 and y_offset > 0:
            preprocessed[
                y_offset:y_offset + height,
                x_offset:x_offset + width,
            ] = rune_box

        predictions = sort_by_confidence(model, preprocessed)
        predictions.sort(key=lambda item: item[1][1])
        classes = [label_map[item[2]] for item in predictions]

        rotated = cv2.rotate(preprocessed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        predictions = sort_by_confidence(model, rotated)
        predictions.sort(key=lambda item: item[1][2], reverse=True)
        rotated_classes = [
            converter[label_map[item[2]]]
            for item in predictions
            if item[2] in (1, 2)
        ]

        for index in range(len(classes)):
            if rotated_classes and classes[index] in ("left", "right"):
                classes[index] = rotated_classes.pop(0)

    return classes


if __name__ == "__main__":
    import mss
    from src.common import config

    if not tensorflow_available():
        raise SystemExit(
            "TensorFlow is not installed. Install requirements-ml.txt before running this test."
        )

    config.enabled = True
    monitor = {"top": 0, "left": 0, "width": 1366, "height": 768}
    model = load_model()
    while True:
        with mss.mss() as sct:
            frame = np.array(sct.grab(monitor))
            cv2.imshow("frame", canny(filter_color(frame)))
            arrows = merge_detection(model, frame)
            print(arrows)
            if cv2.waitKey(1) & 0xFF == 27:
                break
