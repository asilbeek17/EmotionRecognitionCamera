"""
Face emotion recognition: MediaPipe face blendshapes (52 expression scores) + TensorFlow/Keras.

Usage:
  python emotion.py demo                         # instant, rule-based, no training needed
  python emotion.py collect --label happy        # record your own data (do it for each emotion)
  python emotion.py train                        # train a TensorFlow model
  python emotion.py live                         # live recognition with your trained model
"""
import argparse
import collections
import json
import os
import urllib.request

import cv2
import numpy as np

DATA_DIR = "data"
MODEL_DIR = "models"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
KERAS_PATH = os.path.join(MODEL_DIR, "emotion.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "emotion_labels.json")


# --------------------------------------------------------------------------
# Face tracking (MediaPipe Tasks API)
# --------------------------------------------------------------------------
def ensure_face_model():
    if not os.path.exists(FACE_MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading face_landmarker.task (one time, a few MB)...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    return FACE_MODEL_PATH


class FaceTracker:
    def __init__(self):
        import time
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp, self.time = mp, time
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_face_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.t0 = time.time()

    def process(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = int((self.time.time() - self.t0) * 1000)  # must increase every call
        return self.landmarker.detect_for_video(image, ts)


def get_blendshapes(result):
    """Returns (names, scores) of the 52 blendshapes, or (None, None) if no face."""
    if not result.face_blendshapes:
        return None, None
    cats = result.face_blendshapes[0]
    return [c.category_name for c in cats], np.array([c.score for c in cats], np.float32)


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Try --camera 1 "
                           "(on macOS also allow camera access for your Terminal).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    return cap


def process_frame(tracker, frame):
    frame = cv2.flip(frame, 1)  # mirror view (same in collect + live)
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame, tracker.process(rgb)


def draw_face(frame, result):
    h, w = frame.shape[:2]
    for lms in result.face_landmarks:
        for p in lms:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)


def put_text(frame, text, pos, color=(0, 255, 0), scale=0.8):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# --------------------------------------------------------------------------
# Demo: simple rules (no training). Rough, but works instantly.
# --------------------------------------------------------------------------
def rule_based(names, scores):
    d = dict(zip(names, scores))
    smile = (d["mouthSmileLeft"] + d["mouthSmileRight"]) / 2
    frown = (d["mouthFrownLeft"] + d["mouthFrownRight"]) / 2
    brow_down = (d["browDownLeft"] + d["browDownRight"]) / 2
    brow_up = d["browInnerUp"]
    jaw = d["jawOpen"]
    if smile > 0.45:
        return "happy"
    if jaw > 0.35 and brow_up > 0.4:
        return "surprised"
    if brow_down > 0.5:
        return "angry"
    if frown > 0.25 or (brow_up > 0.45 and smile < 0.1):
        return "sad"
    return "neutral"


def demo(args):
    cap, tracker = open_camera(args.camera), FaceTracker()
    recent = collections.deque(maxlen=8)
    print("Press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, result = process_frame(tracker, frame)
        draw_face(frame, result)
        names, scores = get_blendshapes(result)
        if names is None:
            recent.clear()
            put_text(frame, "No face found", (15, 45), (0, 0, 255))
        else:
            recent.append(rule_based(names, scores))
            put_text(frame, "Emotion: " + collections.Counter(recent).most_common(1)[0][0], (15, 45))
        cv2.imshow("emotion demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# 1) Collect your own data
# --------------------------------------------------------------------------
def collect(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{args.label}.npy")
    samples = list(np.load(path)) if os.path.exists(path) else []
    start = len(samples)
    cap, tracker = open_camera(args.camera), FaceTracker()
    recording = False
    print("SPACE = start/stop recording | Q = save & quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, result = process_frame(tracker, frame)
        draw_face(frame, result)
        names, scores = get_blendshapes(result)
        if recording and scores is not None:
            samples.append(scores)
        if recording:
            put_text(frame, "RECORDING", (15, 85), (0, 0, 255), 1.0)
        put_text(frame, f"label='{args.label}' samples={len(samples)} (+{len(samples) - start})", (15, 45))
        cv2.imshow("collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            recording = not recording
    cap.release()
    cv2.destroyAllWindows()
    if samples:
        np.save(path, np.array(samples))
        print(f"Saved {len(samples)} samples to {path}")


# --------------------------------------------------------------------------
# 2) Train with TensorFlow / Keras
# --------------------------------------------------------------------------
def train(args):
    import tensorflow as tf
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    if not os.path.isdir(DATA_DIR):
        raise SystemExit("No data folder. Run the collect command first.")
    labels = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".npy"))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 emotions. Collect more labels.")
    X, y = [], []
    for i, lab in enumerate(labels):
        arr = np.load(os.path.join(DATA_DIR, f"{lab}.npy"))
        print(f"  {lab:12s} {len(arr)} samples")
        X.append(arr)
        y += [i] * len(arr)
    X, y = np.concatenate(X).astype(np.float32), np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    weights = compute_class_weight("balanced", classes=np.arange(len(labels)), y=ytr)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        Xtr, ytr, validation_data=(Xte, yte), epochs=200, batch_size=32,
        class_weight=dict(enumerate(weights)), verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
    )
    print(classification_report(yte, model.predict(Xte, verbose=0).argmax(1), target_names=labels))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(KERAS_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)
    print(f"Saved {KERAS_PATH}")


# --------------------------------------------------------------------------
# 3) Live recognition with your trained model
# --------------------------------------------------------------------------
def live(args):
    import tensorflow as tf

    if not os.path.exists(KERAS_PATH):
        raise SystemExit("No trained model. Run: python emotion.py train")
    model = tf.keras.models.load_model(KERAS_PATH)
    labels = json.load(open(LABELS_PATH))

    cap, tracker = open_camera(args.camera), FaceTracker()
    probs_hist = collections.deque(maxlen=8)  # average over recent frames = steadier output
    print("Press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame, result = process_frame(tracker, frame)
        draw_face(frame, result)
        _, scores = get_blendshapes(result)
        if scores is None:
            probs_hist.clear()
            put_text(frame, "No face found", (15, 45), (0, 0, 255))
        else:
            probs_hist.append(model(scores[None], training=False).numpy()[0])
            probs = np.mean(probs_hist, axis=0)
            top = int(probs.argmax())
            put_text(frame, f"{labels[top]}  {probs[top]:.0%}", (15, 45))
            for i, lab in enumerate(labels):  # probability bars
                y0 = 80 + i * 28
                cv2.rectangle(frame, (15, y0), (15 + int(probs[i] * 200), y0 + 18), (0, 200, 0), -1)
                put_text(frame, lab, (225, y0 + 16), (255, 255, 255), 0.55)
        cv2.imshow("emotion live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Face emotion recognition")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["demo", "collect", "train", "live"]:
        p = sub.add_parser(name)
        p.add_argument("--camera", type=int, default=0)
        if name == "collect":
            p.add_argument("--label", required=True)
    args = ap.parse_args()
    {"demo": demo, "collect": collect, "train": train, "live": live}[args.cmd](args)


if __name__ == "__main__":
    main()
