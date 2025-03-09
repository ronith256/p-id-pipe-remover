import streamlit as st
import cv2
import numpy as np
from yolo.inference import PIDSymbolDetector, PIDPipeRemover
from operations.test import remove_pipes_keep_symbols

st.title("P&ID Diagram Processing")

# Method selection
method = st.selectbox("Select Processing Method", ["YOLO", "Morphological Operations"])

if method == "YOLO":
    st.header("YOLO Parameters")
    model_path = st.text_input("Model Path", "yolo/pid_symbol_detector/yolov11_training/weights/best.pt")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    image_path = st.text_input("Image Path (optional, will be ignored if image is uploaded)", "input.jpg")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    device = st.selectbox("Device", ["cuda", "cpu"])

    if st.button("Process with YOLO"):
        try:
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(image_path)
            detector = PIDSymbolDetector(model_path, conf_threshold, device)
            pipe_remover = PIDPipeRemover()

            results, image = detector.detect_symbols(image)
            symbol_mask = pipe_remover.create_symbol_mask(image, results)
            result_image, debug_info = pipe_remover.remove_pipes(image, symbol_mask)

            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.subheader("Result Image")
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")
elif method == "Morphological Operations":
    st.header("Morphological Operations Parameters")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    image_path = st.text_input("Image Path (optional, will be ignored if image is uploaded)", "input.jpg")
    inverse_method = st.checkbox("Use Inverse Method", False)

    if st.button("Process with Morphological Operations"):
        try:
            if image_file is not None:
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(image_path)
            result, visualization, masks = remove_pipes_keep_symbols(
                image_path,
                use_inverse_method=inverse_method
            )
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.subheader("Result Image")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.subheader("Visualization")
            # Convert matplotlib figure to numpy array
            visualization.canvas.draw()
            img = np.frombuffer(visualization.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(visualization.canvas.get_width_height()[::-1] + (3,))
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")
