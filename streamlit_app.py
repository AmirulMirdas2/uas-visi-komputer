import os
from typing import Tuple, List, Dict, Any

# Force CPU (disable GPU) to avoid CUDA issues if needed
FORCE_CPU = True
if FORCE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ---------------------------------------------------------------------------
# Page config and minimal CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #1e3a8a;
        font-size: 3rem !important;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model & detector loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV Haar Cascade face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


@st.cache_resource
def load_model_and_config() -> Tuple[models.Model, int, List[str]]:
    """
    Build MobileNetV2-based model and try to load weights from known paths.
    Returns (model, img_size, classes).
    """
    IMG_SIZE = 224
    CLASSES = ['with_mask', 'without_mask']
    NUM_CLASSES = len(CLASSES)

    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(name='global_avg_pool'),
            layers.Dense(128, activation='relu', name='dense_128'),
            layers.Dropout(0.5, name='dropout_0.5'),
            layers.Dense(NUM_CLASSES, activation='softmax', name='output'),
        ],
        name='face_mask_model'
    )

    # Try common weight locations
    weight_candidates = [
        "data/models/best_model.h5",
        "data/models/final_model.h5",
        "data/models/model_weights.h5",
    ]
    loaded = False
    for p in weight_candidates:
        if os.path.exists(p):
            try:
                model.load_weights(p, by_name=True, skip_mismatch=True)
                loaded = True
                break
            except Exception:
                continue

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model, IMG_SIZE, CLASSES

# ---------------------------------------------------------------------------
# Detection & classification utilities
# ---------------------------------------------------------------------------

def detect_faces(image: np.ndarray, face_cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    """Detect faces in BGR image; return list of bboxes (x,y,w,h)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def classify_mask(face_img: np.ndarray, model: models.Model, img_size: int) -> Tuple[int, float]:
    """
    Run model prediction on face image (BGR). Returns (class_index, confidence).
    class_index: 0 -> with_mask, 1 -> without_mask
    """
    face_resized = cv2.resize(face_img, (img_size, img_size))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype("float32") / 255.0
    batch = np.expand_dims(face_normalized, axis=0)
    preds = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    return idx, conf


def process_image(image: np.ndarray,
                  model: models.Model,
                  face_cascade: cv2.CascadeClassifier,
                  img_size: int,
                  classes: List[str],
                  confidence_threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect faces, classify mask on each face, draw annotations, and return results.
    If confidence < confidence_threshold, label will be 'uncertain'.
    """
    out = image.copy()
    faces = detect_faces(image, face_cascade)
    results = {'total': len(faces), 'with_mask': 0, 'without_mask': 0, 'uncertain': 0, 'detections': []}

    for (x, y, w, h) in faces:
        if w < 30 or h < 30:
            continue
        face = image[y:y+h, x:x+w]
        if face.size == 0:
            continue

        cls_idx, conf = classify_mask(face, model, img_size)
        label = classes[cls_idx] if conf >= confidence_threshold else 'uncertain'

        if label == 'with_mask':
            results['with_mask'] += 1
            color = (0, 255, 0)
        elif label == 'without_mask':
            results['without_mask'] += 1
            color = (0, 0, 255)
        else:
            results['uncertain'] += 1
            color = (0, 255, 255)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{label} ({conf*100:.1f}%)" if label != 'uncertain' else f"uncertain ({conf*100:.1f}%)"
        cv2.putText(out, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        results['detections'].append({
            'label': label,
            'confidence': float(conf),
            'bbox': (int(x), int(y), int(w), int(h))
        })

    return out, results

# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown("<h1 class='stTitle'>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading model and face detector..."):
        model, img_size, classes = load_model_and_config()
        face_cascade = load_face_detector()

    if model is None:
        st.error("Failed to load model.")
        return

    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.selectbox("Mode", ["📸 Image Upload", "🎥 Video Upload", "ℹ️ About"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info**")
    st.sidebar.info(f"Architecture: MobileNetV2\nInput: {img_size}×{img_size}\nClasses: {', '.join(classes)}")

    # ----- IMAGE MODE -----
    if mode == "📸 Image Upload":
        st.header("Upload Image")
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Unable to read image.")
                return

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

            with st.spinner("Processing..."):
                processed, results = process_image(image, model, face_cascade, img_size, classes, confidence_threshold)

            with col2:
                st.subheader("Result")
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.markdown("---")
            st.subheader("Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Faces", results['total'])
            c2.metric("✅ With Mask", results['with_mask'])
            c3.metric("❌ Without Mask", results['without_mask'])
            c4.metric("⚠️ Uncertain", results['uncertain'])

            if results['detections']:
                df = pd.DataFrame(results['detections'])
                df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(df[['label', 'confidence', 'bbox']], use_container_width=True)

            if results['without_mask'] > 0:
                st.error(f"{results['without_mask']} person(s) detected without mask.")
            elif results['with_mask'] > 0 and results['without_mask'] == 0:
                st.success("All detected faces are wearing masks.")
            else:
                st.info("No faces detected.")

    # ----- VIDEO MODE -----
    elif mode == "🎥 Video Upload":
        st.header("Upload Video")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded_video.read())
            tmp.flush()
            tmp.close()

            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                st.error("Unable to open video.")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            st.write(f"FPS: {fps} | Frames: {frames}")

            step = st.slider("Process every N frames (speed)", 1, 10, 3)
            start_btn = st.button("Start Processing")

            if start_btn:
                progress = st.progress(0)
                status = st.empty()
                frame_idx = 0
                processed_idx = 0
                stats = {'total_faces': 0, 'with_mask': 0, 'without_mask': 0, 'uncertain': 0}
                frame_placeholder = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx % step != 0:
                        continue
                    processed_idx += 1
                    processed_frame, r = process_image(frame, model, face_cascade, img_size, classes, confidence_threshold)

                    stats['total_faces'] += r.get('total', 0)
                    stats['with_mask'] += r.get('with_mask', 0)
                    stats['without_mask'] += r.get('without_mask', 0)
                    stats['uncertain'] += r.get('uncertain', 0)

                    if processed_idx % 30 == 0:
                        frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}", use_column_width=True)

                    if frames:
                        progress.progress(min(1.0, frame_idx / frames))
                    status.text(f"Processed frame {frame_idx}")

                cap.release()
                st.success("Video processing complete!")
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Faces", stats['total_faces'])
                c2.metric("✅ With Mask", stats['with_mask'])
                c3.metric("❌ Without Mask", stats['without_mask'])
                c4.metric("⚠️ Uncertain", stats['uncertain'])

    # ----- ABOUT -----
    else:
        st.header("About")
        st.markdown(
            """
            **Face Mask Detection** built with MobileNetV2 + OpenCV Haar Cascade.
            Upload images or videos to detect faces and classify mask usage.
            """
        )

if __name__ == "__main__":
    main()