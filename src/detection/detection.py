"""Directional-arrow classification with optional TensorFlow support."""

from __future__ import annotations

import cv2
import numpy as np

from src.common import utils

try:
    import tensorflow as tf
except ImportError:
    tf = None


class TensorFlowUnavailableError(RuntimeError):
    """Raised when Rune detection is requested without TensorFlow installed."""


def tensorflow_available() -> bool:
    """Return whether the optional TensorFlow runtime is available."""
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
