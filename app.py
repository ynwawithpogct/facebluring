from flask import Flask, render_template, request, Response, send_file, redirect, url_for
import os
from utils.video_processing import process_video, generate_video_stream, add_face_to_db
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
FACES_FOLDER = os.path.join(UPLOAD_FOLDER, 'faces')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['FACES_FOLDER'] = FACES_FOLDER

MODEL = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet"
]
BACKEND = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yolov11s',
  'yolov11n',
  'yolov11m',
  'yunet',
  'centerface',
]
BLUR_TYPE = [
  'pixelation', 
  'gaussian blur',
]

app.config['MODEL'] = MODEL
app.config['BACKEND'] = BACKEND
app.config['BLUR_TYPE'] = BLUR_TYPE

FACE_FILENAME_LIST = [file for file in os.listdir(app.config['FACES_FOLDER'])]

model_name = app.config['MODEL'][6]
detector_backend = app.config['BACKEND'][4]
blur_name = app.config['BLUR_TYPE'][1]
face_filename = FACE_FILENAME_LIST[0] if len(FACE_FILENAME_LIST) > 0 else None

@app.route('/')
def index():
    return render_template('index.html', models=app.config['MODEL'], model_name=model_name, backend=app.config['BACKEND'], detector_backend=detector_backend, face_filename_list=FACE_FILENAME_LIST, face_filename=face_filename, blur_type_list=app.config['BLUR_TYPE'], blur_name=blur_name)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video uploaded!", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected file!", 400

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    # Process video with DeepFace
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{model_name}_{detector_backend}_{blur_name}_{filename}")
    if face_filename is not None:
        face_path = os.path.join(app.config['FACES_FOLDER'], face_filename)
        process_video(video_path, output_path, face_path, model_name, detector_backend, blur_name)

        return render_template('video_stream.html', video_path=output_path)
    else:
        return "No face filename chosen uploaded!", 300

@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'face' not in request.files:
        return "No face image uploaded!", 400

    face = request.files['face']
    if face.filename == '':
        return "No selected file!", 400

    filename = secure_filename(face.filename)
    face_path = os.path.join(app.config['FACES_FOLDER'], filename)
    face.save(face_path)

    # Add face to database
    added = add_face_to_db(face_path, model_name, detector_backend)
    global FACE_FILENAME_LIST
    if added and filename not in FACE_FILENAME_LIST:
        FACE_FILENAME_LIST.append(filename)

    return redirect(url_for('index'))

@app.route('/stream/<filename>')
def stream_video(filename):
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download/<filename>')
def download_video(filename):
    video_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_file(video_path, as_attachment=True)

@app.route('/select_model', methods=['POST'])
def select_model():
    # Lấy giá trị được lựa chọn từ form
    global model_name
    model_name = request.form.get('model')
    # In ra giá trị được chọn
    print(f'Selected model: {model_name}')
    
    # Bạn có thể lưu giá trị này vào biến toàn cục hoặc xử lý nó theo cách bạn muốn
    return redirect(url_for('index'))

@app.route('/select_detector_backend', methods=['POST'])
def select_detector_backend():
    # Lấy giá trị được lựa chọn từ form
    global detector_backend
    detector_backend = request.form.get('detector_backend')
    # In ra giá trị được chọn
    print(f'Selected detector backend: {detector_backend}')
    
    # Bạn có thể lưu giá trị này vào biến toàn cục hoặc xử lý nó theo cách bạn muốn
    return redirect(url_for('index'))

@app.route('/select_face_filename', methods=['POST'])
def select_face_filename():
    # Lấy giá trị được lựa chọn từ form
    global face_filename, FACE_FILENAME_LIST
    face_filename = request.form.get('face_filename')
    # In ra giá trị được chọn
    print(f'Selected face filename: {face_filename}')
    
    # Bạn có thể lưu giá trị này vào biến toàn cục hoặc xử lý nó theo cách bạn muốn
    return redirect(url_for('index'))

@app.route('/select_blur_type', methods=['POST'])
def select_blur_type():
    # Lấy giá trị được lựa chọn từ form
    global blur_name
    blur_name = request.form.get('blur_type')
    # In ra giá trị được chọn
    print(f'Selected blur type: {blur_name}')
    
    # Bạn có thể lưu giá trị này vào biến toàn cục hoặc xử lý nó theo cách bạn muốn
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)