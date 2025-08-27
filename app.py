from flask import Flask, Response, render_template, jsonify, send_file, request, redirect, url_for
from ultralytics import YOLO
import cv2
import easyocr
import sqlite3
from datetime import datetime
import re
import pytesseract
import numpy as np
import csv
import os
import time
from scipy.signal import convolve2d
import scipy.signal
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import threading
import torch

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    
    model_name = 'RealESRGAN_x4plus'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        os.makedirs('weights', exist_ok=True)
        load_file_from_url(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model_dir='weights')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        device=device)
    has_esrgan = True
except ImportError:
    print("ESRGAN not available. Install with: pip install realesrgan basicsr")
    has_esrgan = False

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

model = YOLO('best.pt')
easy_reader = easyocr.Reader(['en'])

processing_status = {
    'is_processing': False,
    'progress': 0,
    'total_frames': 0,
    'processed_frames': 0,
    'plates_found': 0,
    'current_file': '',
    'plates': []
}

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS plates
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate TEXT,
                  timestamp TEXT,
                  source TEXT,
                  frame_number INTEGER DEFAULT 0,
                  confidence REAL DEFAULT 0.0)''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_threshold(gray_img):
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def segment_characters(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 15 < h < 150 and 10 < w < 80:
            char_regions.append((x, y, w, h))
    char_regions = sorted(char_regions, key=lambda b: b[0])
    chars = []
    for x, y, w, h in char_regions:
        char = thresh_img[y:y+h, x:x+w]
        char = cv2.resize(char, (40, 60))
        chars.append(char)
    return chars

def tesseract_read(thresh_img):
    chars = segment_characters(thresh_img)
    plate_text = ""
    for char_img in chars:
        char_text = pytesseract.image_to_string(char_img, config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        char_text = re.sub(r'[^A-Za-z0-9]', '', char_text)
        plate_text += char_text
    return plate_text

def is_blurry(image):
    return cv2.Laplacian(image, cv2.CV_64F).var() < 100

def wiener_deblur(image, kernel_size=5, noise_power=0.01):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    deblurred = scipy.signal.wiener(image, mysize=kernel.shape, noise=noise_power)
    return np.uint8(np.clip(deblurred, 0, 255))

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=10):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    low_contrast_mask = np.absolute(image - blurred) < threshold
    np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def enhance_with_esrgan(image):
    if not has_esrgan:
        return image
    
    try:
        if len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        output, _ = upsampler.enhance(image, outscale=2)
        
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
    except Exception as e:
        print(f"ESRGAN enhancement failed: {e}")
        return image

def enhance_blurry_plate(image):
    if has_esrgan and is_blurry(image):
        enhanced = enhance_with_esrgan(image)
        if enhanced is not None:
            return enhanced

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    deblurred = wiener_deblur(gray)
    enhanced = unsharp_mask(deblurred)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def clean_plate_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if 6 <= len(text) <= 12:
        return text
    return "Invalid Format"

def correct_plate_format(plate_text):
    """Corrects common OCR errors in license plates without using external APIs"""
    if not plate_text or plate_text == "Invalid Format":
        return plate_text

    ocr_corrections = {
        '0': 'O', 'O': '0', 
        '1': 'I', 'I': '1',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        '2': 'Z', 'Z': '2',
        '6': 'G', 'G': '6',
        'D': '0', 'Q': '0'
    }
    
    cleaned = ""
    for i, char in enumerate(plate_text):
        if i < 2:
            if char in '0123456789':
                if char in ocr_corrections:
                    char = ocr_corrections[char]
        elif i < 4:
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if char in ocr_corrections:
                    char = ocr_corrections[char]
        elif i < 6:
            if char in '0123456789':
                if char in ocr_corrections:
                    char = ocr_corrections[char]
        else:
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if char in ocr_corrections:
                    char = ocr_corrections[char]
        
        cleaned += char
    
    if len(cleaned) >= 4 and cleaned[2:4] == 'BH':
        first_two = cleaned[:2]
        for i, char in enumerate(first_two):
            if char not in '0123456789':
                if char in ocr_corrections:
                    first_two = first_two[:i] + ocr_corrections[char] + first_two[i+1:]
        cleaned = first_two + cleaned[2:]
    
    return cleaned

def process_plate(image, confidence=0.0):
    original_image = image.copy()
    
    if is_blurry(image):
        image = enhance_blurry_plate(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = apply_threshold(gray)
    plate_text = tesseract_read(thresh)

    if len(plate_text) < 6:
        result = easy_reader.readtext(image)
        if result:
            plate_text = result[0][1]

    plate_text = clean_plate_text(plate_text)

    if plate_text != "Invalid Format":
        plate_text = correct_plate_format(plate_text)

    return plate_text, confidence

def process_video_file(file_path):
    global processing_status
    
    processing_status['is_processing'] = True
    processing_status['progress'] = 0
    processing_status['plates'] = []
    processing_status['plates_found'] = 0
    processing_status['current_file'] = os.path.basename(file_path)
    
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    processing_status['total_frames'] = total_frames
    processing_status['processed_frames'] = 0
    
    frame_interval = 5
    frame_number = 0
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_number % frame_interval == 0:
                results = model.predict(frame, conf=0.5)[0]
                
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    x1, y1 = max(0, x1-10), max(0, y1-10)
                    x2, y2 = min(frame.shape[1], x2+10), min(frame.shape[0], y2+10)
                    
                    if x1 < x2 and y1 < y2:  # Valid box dimensions
                        cropped = frame[y1:y2, x1:x2]
                        
                        if cropped.size > 0:  # Non-empty image
                            plate_text, conf = process_plate(cropped, confidence)
                            
                            if plate_text != "Invalid Format":
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                source = os.path.basename(file_path)
                                
                                # Save to database
                                c.execute(
                                    "INSERT INTO plates (plate, timestamp, source, frame_number, confidence) VALUES (?, ?, ?, ?, ?)",
                                    (plate_text, timestamp, source, frame_number, confidence)
                                )
                                conn.commit()
                                
                                # Update processing status
                                processing_status['plates_found'] += 1
                                processing_status['plates'].append({
                                    'plate': plate_text,
                                    'frame': frame_number,
                                    'timestamp': timestamp,
                                    'confidence': round(confidence * 100, 2)
                                })
            
            frame_number += 1
            processing_status['processed_frames'] = frame_number
            processing_status['progress'] = int((frame_number / total_frames) * 100)
    
    except Exception as e:
        print(f"Error processing video: {e}")
    
    finally:
        cap.release()
        conn.close()
        processing_status['is_processing'] = False
        
        if os.path.exists(file_path):
            os.remove(file_path)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model.predict(frame, conf=0.5)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            x1, y1 = max(0, x1-10), max(0, y1-10)
            x2, y2 = min(frame.shape[1], x2+10), min(frame.shape[0], y2+10)
            cropped = frame[y1:y2, x1:x2]
            
            plate_text, conf = process_plate(cropped, confidence)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            if plate_text != "Invalid Format":
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute(
                    "INSERT INTO plates (plate, timestamp, source, confidence) VALUES (?, ?, ?, ?)",
                    (plate_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'webcam', confidence)
                )
                conn.commit()
                conn.close()
            break
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        thread = threading.Thread(target=process_video_file, args=(file_path,))
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('processing_status_page'))
    
    return redirect(url_for('index'))

@app.route('/processing')
def processing_status_page():
    return render_template('processing.html')

@app.route('/status')
def status():
    global processing_status
    return jsonify(processing_status)

@app.route('/latest_plate')
def latest_plate():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT plate FROM plates ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return jsonify({'plate': row[0] if row else ""})

@app.route('/latest_plates')
def latest_plates():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT plate, timestamp, source, confidence FROM plates ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    results = []
    for row in rows:
        results.append({
            'plate': row[0],
            'timestamp': row[1],
            'source': row[2],
            'confidence': row[3]
        })
    conn.close()
    return jsonify(results)

@app.route('/export_csv')
def export_csv():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT id, plate, timestamp, source, frame_number, confidence FROM plates")
    rows = c.fetchall()
    conn.close()
    
    filename = f'plates_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Plate', 'Timestamp', 'Source', 'Frame Number', 'Confidence'])
        writer.writerows(rows)
    
    return send_file(filename, as_attachment=True)

@app.route('/search')
def search():
    query = request.args.get('q', '').upper()
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute(
        "SELECT plate, timestamp, source, confidence FROM plates WHERE plate LIKE ? ORDER BY timestamp DESC", 
        ('%' + query + '%',)
    )
    results = c.fetchall()
    formatted_results = []
    for row in results:
        formatted_results.append({
            'plate': row[0],
            'timestamp': row[1],
            'source': row[2],
            'confidence': row[3]
        })
    conn.close()
    return jsonify(formatted_results)

@app.route('/stats')
def stats():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM plates")
    total_count = c.fetchone()[0]
    
    c.execute("SELECT source, COUNT(*) FROM plates GROUP BY source")
    sources = c.fetchall()
    
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute("SELECT COUNT(*) FROM plates WHERE timestamp LIKE ?", (today + '%',))
    today_count = c.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_plates': total_count,
        'today_plates': today_count,
        'sources': dict(sources)
    })

@app.route('/clear_database')
def clear_database():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("DELETE FROM plates")
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)