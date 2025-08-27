IPD – License Plate Detection
Features

Detect license plates from webcam or uploaded videos

Uses YOLOv8 + EasyOCR + Tesseract

Enhances blurry plates with ESRGAN / OpenCV

Saves results in SQLite database

Export plates to CSV

Installation
git clone https://github.com/AMAANL/ipd.git
cd ipd
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt


Install Tesseract OCR:

macOS: brew install tesseract

Ubuntu: sudo apt install tesseract-ocr

Windows: Download here

Model Weights

Place your YOLO model file in the weights/ folder:

weights/best.pt

Usage

Run the app:

python app.py


Open in browser:

http://127.0.0.1:5000/

API Endpoints

/ → Home

/video_feed → Webcam detection

/upload → Upload video

/status → Processing status (JSON)

/latest_plate → Last detected plate

/latest_plates → 10 latest detections

/export_csv → Export results as CSV

Notes

.pt weights are not uploaded to GitHub (too large).

database.db and uploads/ are generated locally.
