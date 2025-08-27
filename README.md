# IPD – License Plate Detection

## Features
### Detect license plates from webcam or uploaded videos  
### YOLOv8 + EasyOCR + Tesseract  
### ESRGAN + OpenCV for blurry plate enhancement  
### SQLite database for storing results  
### Export detections to CSV  

---

## Installation
### Clone repo
```bash
git clone https://github.com/AMAANL/ipd.git
cd ipd
# Create virtual environment

## Clone repo
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
## Install requirements
```bash
pip install -r requirements.txt
##Place YOLO weights in weights/
```bash
weights/best.pt
