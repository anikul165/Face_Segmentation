import torch
import torchvision.transforms as T
import numpy as np
import cv2
import streamlit as st
import mediapipe as mp
from PIL import Image
import os

class FaceHairSegmenter:
    def __init__(self):
        # Use MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Use full range model
            min_detection_confidence=0.6
        )
        
        # Load BiSeNet model
        self.model = self.load_model()
        
        # Define transforms - adjust according to BiSeNet requirements
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CelebAMask-HQ classes - focus on the categories we want to keep
        self.keep_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]  # All except 0, 14, 16

    def load_model(self):
        try:
            # Import locally to avoid dependency issues if model isn't present
            from model import BiSeNet
            
            # Initialize BiSeNet with 19 classes (for CelebAMask-HQ)
            model = BiSeNet(n_classes=19)
            
            # Load the pretrained weights
            model.load_state_dict(torch.load('bisenet.pth', map_location=torch.device('cpu')))
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
                
            print("BiSeNet model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def detect_faces(self, image):
        """Detect faces using MediaPipe (expects image in RGB)."""
        # Since image from cv2 is BGR, convert to RGB for MediaPipe
        image_rgb = image if len(image.shape) == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process with MediaPipe
        results = self.face_detection.process(image_rgb)
        
        bboxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = max(0, int(bbox.xmin * w))
                y_min = max(0, int(bbox.ymin * h))
                x_max = min(w, int((bbox.xmin + bbox.width) * w))
                y_max = min(h, int((bbox.ymin + bbox.height) * h))
                bboxes.append((x_min, y_min, x_max, y_max))
        
        if len(bboxes) > 1:
            bboxes = self.remove_overlapping_boxes(bboxes)
            
        return len(bboxes), bboxes

    def remove_overlapping_boxes(self, boxes, overlap_threshold=0.5):
        if not boxes:
            return []
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        boxes = sorted(boxes, key=box_area, reverse=True)
        keep = []
        for current in boxes:
            is_duplicate = False
            for kept_box in keep:
                x1 = max(current[0], kept_box[0])
                y1 = max(current[1], kept_box[1])
                x2 = min(current[2], kept_box[2])
                y2 = min(current[3], kept_box[3])
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = box_area(current)
                    area2 = box_area(kept_box)
                    union = area1 + area2 - intersection
                    iou = intersection / union
                    if iou > overlap_threshold:
                        is_duplicate = True
                        break
            if not is_duplicate:
                keep.append(current)
        return keep

    def segment_face_hair(self, image):
        """Segment face using BiSeNet trained on CelebAMask-HQ."""
        if self.model is None:
            return image, "Model not loaded correctly."
        if image is None or image.size == 0:
            return image, "Invalid image provided."
        
        # Detect faces
        num_faces, bboxes = self.detect_faces(image)
        if num_faces == 0:
            return image, "No face detected! Please upload an image with a clear face."
        elif num_faces > 1:
            debug_img = image.copy()
            for (x_min, y_min, x_max, y_max) in bboxes:
                cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            return debug_img, f"{num_faces} faces detected! Please upload an image with exactly ONE face."
        
        # Get the face bounding box (we'll use this only for ROI, not for final segmentation)
        bbox = bboxes[0]
        x_min, y_min, x_max, y_max = bbox
        h, w = image.shape[:2]
        
        # Expand bounding box for better segmentation
        face_height = y_max - y_min + 550
        face_width = x_max - x_min + 550
        
        y_min_exp = max(0, y_min - int(face_height * 0.5))  # Expand more for hair
        x_min_exp = max(0, x_min - int(face_width * 0.3))
        x_max_exp = min(w, x_max + int(face_width * 0.3))
        y_max_exp = min(h, y_max + int(face_height * 0.2))
        
        # Crop and prepare image for BiSeNet
        face_region = image[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
        original_face_size = face_region.shape[:2]
        
        # Ensure RGB format for PIL
        if face_region.shape[2] == 3:
            pil_face = Image.fromarray(face_region)
        else:
            pil_face = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
            
        # Apply transformations and run model
        input_tensor = self.transform(pil_face).unsqueeze(0)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        with torch.no_grad():
            out = self.model(input_tensor)[0]
            parsing = out.squeeze(0).argmax(0).byte().cpu().numpy()
        
        # Resize parsing map back to original size
        parsing = cv2.resize(parsing, (original_face_size[1], original_face_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Create mask that keeps only the classes we want
        mask = np.zeros_like(parsing, dtype=np.uint8)
        for cls_id in self.keep_classes:
            mask[parsing == cls_id] = 255
            
        # Refine the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create full image mask (initialize with zeros)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        # Place the face mask in the right position
        full_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = mask
        
        # Create the RGBA output
        if image.shape[2] == 3:  # RGB
            rgba = np.dstack((image, np.zeros((h, w), dtype=np.uint8)))
            # Copy only the face region with its alpha
            rgba[y_min_exp:y_max_exp, x_min_exp:x_max_exp, 3] = mask
        else:  # Already RGBA or other format
            rgba = np.dstack((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                             np.zeros((h, w), dtype=np.uint8)))
            rgba[y_min_exp:y_max_exp, x_min_exp:x_max_exp, 3] = mask
            
        return rgba, "Face segmented successfully!"

# Streamlit app
def main():
    st.set_page_config(page_title="Face Segmentation Tool", layout="wide")
    
    st.title("Face Segmentation Tool")
    st.markdown("""
    Upload an image to extract the face with a transparent background.
    
    ## Guidelines:
    - Upload an image with **exactly one face**
    - The face should be clearly visible
    - For best results, use images with good lighting
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert to numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Segment Face"):
                with st.spinner("Processing..."):
                    segmenter = FaceHairSegmenter()
                    result, message = segmenter.segment_face_hair(image)
                    
                    with col2:
                        st.header("Segmented Result")
                        st.image(result, caption="Segmented Face", use_container_width=True)
                        st.text(message)
                        
                        # Add download button for the result
                        if "No face detected" not in message and "faces detected" not in message:
                            # Convert numpy array to PIL Image
                            result_img = Image.fromarray(result)
                            
                            # Create a BytesIO object
                            from io import BytesIO
                            buf = BytesIO()
                            result_img.save(buf, format="PNG")
                            
                            # Add download button
                            st.download_button(
                                label="Download Segmented Face",
                                data=buf.getvalue(),
                                file_name="segmented_face.png",
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()