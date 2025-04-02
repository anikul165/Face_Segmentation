# Face Segmentation Tool

A deep learning-based tool for precise face segmentation using BiSeNet trained on the CelebAMask-HQ dataset. This tool extracts faces from images with a transparent background, perfect for creating profile pictures, avatars, or creative photo editing.

## Features

- Accurate face and hair segmentation with transparent background
- 19-class facial attribute segmentation (skin, eyes, eyebrows, nose, lips, hair, etc.)
- User-friendly Streamlit web interface
- MediaPipe face detection to identify and focus on faces
- Support for downloading the segmented result

## Technical Details

This project uses:

- **BiSeNet** (Bilateral Segmentation Network) for semantic segmentation
- **CelebAMask-HQ dataset** trained model with 19 facial attribute classes
- **MediaPipe** for initial face detection and bounding box estimation
- **PyTorch** for the deep learning components
- **Streamlit** for the web interface

## Installation

### Prerequisites

- Python 3.7 or newer
- CUDA-compatible GPU (optional, but recommended for faster processing)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/face-segmentation.git
   cd face-segmentation
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the BiSeNet model weights file (`bisenet.pth`) in the project root directory.
   (The file should already be included in the repository)

## Usage

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to the URL shown in the console (typically http://localhost:8501)

3. Upload an image with a face

4. Click the "Segment Face" button

5. View and download the segmented result

## Class Labels in CelebAMask-HQ

The model recognizes 19 different facial attributes:

| ID  | Class      | Description               |
| --- | ---------- | ------------------------- |
| 0   | background | Non-face background areas |
| 1   | skin       | Face skin                 |
| 2   | nose       | Nose                      |
| 3   | eye_g      | Eyeglasses                |
| 4   | l_eye      | Left eye                  |
| 5   | r_eye      | Right eye                 |
| 6   | l_brow     | Left eyebrow              |
| 7   | r_brow     | Right eyebrow             |
| 8   | l_ear      | Left ear                  |
| 9   | r_ear      | Right ear                 |
| 10  | mouth      | Mouth                     |
| 11  | u_lip      | Upper lip                 |
| 12  | l_lip      | Lower lip                 |
| 13  | hair       | Hair                      |
| 14  | hat        | Hat                       |
| 15  | ear_r      | Ear rings                 |
| 16  | neck_l     | Neck area                 |
| 17  | neck       | Neck                      |
| 18  | cloth      | Clothing                  |

## How It Works

1. The app uses MediaPipe to detect faces in the uploaded image
2. It crops and processes the face region using BiSeNet
3. BiSeNet performs semantic segmentation to classify each pixel into one of 19 classes
4. Selected facial features are preserved while background, neck, and clothes are made transparent
5. The result is an RGBA image with the face and hair intact and a transparent background

## Customization

You can modify which facial attributes to keep by adjusting the `keep_classes` list in the `FaceHairSegmenter` class:

```python
# Current configuration - keeps all face parts except background, clothes, and neck
self.keep_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18]
```

## Troubleshooting

- **No face detected**: Ensure the image contains a clearly visible face.
- **Multiple faces detected**: The app works best with a single face per image.
- **Poor segmentation**: For best results, use images with good lighting and a clear face.
- **CUDA out of memory**: Try using a smaller image or run on CPU if your GPU has limited memory.

## References

- [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)
- [CelebAMask-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
