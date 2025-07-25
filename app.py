import os
import numpy as np
import cv2 as cv
import pickle
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package='Custom', name='L2NormLayer')
class L2NormLayer(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0,
                                       min_detection_confidence=0.5)

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_embedding_model = load_model(
    "face_embedding_model.keras",
    custom_objects={"L2NormLayer": L2NormLayer}
)
with open("data_embedding.pkl", "rb") as f:
    data_embedding = pickle.load(f)

def find(embedding):
    mx, best = -1, "Unknown"
    for celeb, emb in data_embedding.items():
        sim = cosine_similarity([embedding], [emb])[0][0]
        if sim > mx:
            mx, best = sim, celeb
    return mx, best

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    face_filename = None

    if request.method == 'POST':
        upload = request.files['image']
        filename = secure_filename(upload.filename)

        data = upload.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            result = "Invalid image"
        else:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            dets = face_detection.process(img_rgb)
            if not dets.detections:
                result = "No face detected"
            else:
                d = dets.detections[0].location_data.relative_bounding_box
                h, w, _ = img.shape
                x1 = max(int(d.xmin * w), 0)
                y1 = max(int(d.ymin * h), 0)
                x2 = min(w, x1 + int(d.width * w))
                y2 = min(h, y1 + int(d.height * h))
                face = img[y1:y2, x1:x2]
                face = cv.resize(face, (256, 256))

                face_filename = f"face_{filename}"
                cv.imwrite(os.path.join(UPLOAD_FOLDER, face_filename), face)

                raw = face_embedding_model.predict(
                    np.expand_dims(face, axis=0), verbose=0
                )
                emb = raw[0]
                score, match = find(emb)
                result = f"WOW you look like {match}! (Similarity: {score:.2f})"

    return render_template('index.html',
                           result=result,
                           face_filename=face_filename)

if __name__ == '__main__':
    app.run(debug=True)
