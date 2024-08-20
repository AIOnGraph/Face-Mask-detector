import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from utils import get_ice_servers
from av import VideoFrame

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.11
NMS_THRESHOLD = 0.45
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Text parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0, 0, 0)
BLUE   = (255, 178, 50)
YELLOW = (0, 255, 255)

class VideoProcessor1(VideoProcessorBase):
    def __init__(self, confidence_threshold, score_threshold):
        modelWeights = "last.onnx"
        self.net = cv2.dnn.readNet(modelWeights)
        self.confidence_threshold = confidence_threshold
        self.score_threshold = score_threshold

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        detections = pre_process(image, self.net)
        processed_image = post_process(image, detections, self.confidence_threshold, self.score_threshold)
        return VideoFrame.from_ndarray(processed_image, format="bgr24")


def load_classes():
    with open("coco.names", 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

def draw_label(im, label, x, y):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process(input_image, outputs, confidence_threshold, score_threshold):
    classes = load_classes()
    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= confidence_threshold:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > score_threshold:
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_THRESHOLD)
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            left, top, width, height = box
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
            draw_label(input_image, label, left, top)
    return input_image


st.title("Face Mask Detection")
with st.sidebar:
    confidence_threshold = st.slider("**Confidence Threshold**", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.01, key="confidence__threshold")
    score_threshold = st.slider("**Score Threshold**", 0.0, 1.0, SCORE_THRESHOLD, 0.01, key="score__threshold")
    mode = st.radio("**Choose mode**", ("Webcam", "Image Upload"))

if mode == "Webcam":
    webrtc_streamer(key="face_mask", mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": get_ice_servers(),
            "iceTransportPolicy": "relay",
        },
        media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
        video_processor_factory=lambda: VideoProcessor1(confidence_threshold, score_threshold),
        async_processing=True
    )

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        net = cv2.dnn.readNet("last.onnx")
        detections = pre_process(image, net)
        processed_image = post_process(image, detections, st.session_state.confidence__threshold, st.session_state.score__threshold)
        st.image(processed_image, channels="BGR")
