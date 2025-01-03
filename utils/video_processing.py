import cv2
from deepface import DeepFace
import pandas as pd
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
# import ast

# Tạo một DataFrame lưu khuôn mặt
FACE_DB = None

# Đường dẫn đến tệp hiện tại (hoặc thư mục hiện tại)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến thư mục cha
parent_dir = os.path.dirname(current_dir)

# Đường dẫn đến thư mục Face_DB
face_dataframe = os.path.join(parent_dir, 'dataframe', 'Face_DB.csv')

# Kiểm tra đã tồn tại chưa
if os.path.exists(face_dataframe):
    # tải FACE_DB về
    FACE_DB = pd.read_csv(face_dataframe)
    print("Tệp Face_DB.csv đã được tải thành công")
    # FACE_DB["embedding"] = FACE_DB["embedding"].apply(lambda x: ast.literal_eval(x)).astype(np.float32)
    #print(FACE_DB["embedding"])
    FACE_DB["embedding"] = FACE_DB["embedding"].apply(lambda x: np.fromstring(' '.join(x.strip("[]").split('\n')), sep=" ", dtype=np.float32))
    #print((FACE_DB["embedding"][0].shape))
else:
    #Tạo một DataFrame mới
    print("Tệp Face_DB.csv không tồn tại tại.")
    FACE_DB = pd.DataFrame(columns=["identity", "model_name", "detector_backend", "number", "embedding"])

def add_face_to_db(face_path, model_name, detector_backend):
    """
    Tải ảnh khuôn mặt lên DataFrame làm cơ sở nhận diện.
    """
    # kiểm tra khuôn mặt đã có trong data base chưa
    global FACE_DB
    target_face = FACE_DB.loc[(FACE_DB['identity'] == face_path) & (FACE_DB['model_name'] == model_name) & (FACE_DB['detector_backend'] == detector_backend)]
    if not target_face.empty:
        print('face đã có trong dataframe')
        return False
    # Bắt đầu tải ảnh 
    print('Bắt đầu tải ảnh...')
    n = 0
    face_embeddings = DeepFace.represent(face_path, model_name=model_name, detector_backend=detector_backend)
    for face in face_embeddings:
        embedding = face["embedding"]
        #print(face)
        df = pd.DataFrame({
                        "identity": face_path, 
                        "model_name": model_name, 
                        "detector_backend": detector_backend, 
                        "number": n, 
                        "embedding": [embedding]
                    })
        df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
        #print(df["embedding"][0].shape)
        FACE_DB = pd.concat([
                             FACE_DB, 
                             df
                             ],
                            ignore_index=True
                            )
        n += 1
    #print(FACE_DB)
    FACE_DB.to_csv(face_dataframe, index=False)
    return True


def pixelate(image, box, pixel_size=10):
    """
    Thực hiện pixelation cho một vùng trong ảnh.
    
    Args:
    - image: ảnh gốc.
    - box: bounding box dưới dạng (x, y, w, h).
    - pixel_size: kích thước của pixel (mỗi pixel là một ô vuông, mặc định là 10).
    
    Returns:
    - ảnh sau khi pixelation.
    """
    (x, y, w, h) = box
    # Cắt vùng trong bounding box
    roi = image[y:y+h, x:x+w]
    
    # Dùng hàm resize để giảm độ phân giải của ảnh (giống như pixel hóa)
    roi_resized = cv2.resize(roi, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    
    # Resize lại ảnh về kích thước gốc của bounding box
    pixelated_roi = cv2.resize(roi_resized, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Thay thế lại vùng ảnh cũ với phần ảnh đã pixel hóa
    image[y:y+h, x:x+w] = pixelated_roi
    return image

def process_video(input_path, output_path, face_path, model_name, detector_backend, blur_name):
    """
    Xử lý video
    """
    # kiểm tra khuôn mặt đã có trong data base chưa
    global FACE_DB
    target_face = FACE_DB.loc[(FACE_DB['identity'] == face_path) & (FACE_DB['model_name'] == model_name) & (FACE_DB['detector_backend'] == detector_backend)]
    if target_face.empty:
        print("Chưa có face trong dataframe")
        add_face_to_db(face_path=face_path, model_name=model_name, detector_backend=detector_backend)
        target_face = FACE_DB.loc[(FACE_DB['identity'] == face_path) & (FACE_DB['model_name'] == model_name) & (FACE_DB['detector_backend'] == detector_backend)]
    # Bắt đầu sử lý videos
    print("Bắt đầu tạo videos")
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"FPS của video là: {fps}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    tracker = DeepSort(max_age=30)
    print("Bắt đầu sử lý videos")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sử dụng DeepFace để detect khuôn mặt
        try:
            # Dùng extract_faces thay vì detectFace
            faces = DeepFace.extract_faces(frame, detector_backend=detector_backend, enforce_detection=False)
            #print('detectFace xong')
            # print(len(faces))
            detect = []
            for face in faces:
                # Kiểm tra nếu khuôn mặt này khớp với database
                # print(type(face))
                # print(face['face'].shape)
                embedding = DeepFace.represent(face['face'], model_name=model_name, detector_backend='skip')[0]["embedding"]
                for _, row in target_face.iterrows():
                    # print(row["embedding"].shape)
                    #distance = np.linalg.norm(np.array(row["embedding"]) - np.array(embedding))
                    # Tính Cosine Similarity
                    embedding1 = row["embedding"]
                    embedding2 = np.array(embedding, dtype=np.float32)
                    # print(embedding1.shape, embedding2.shape)
                    # print(np.dot(embedding1, embedding2), np.linalg.norm(embedding1), np.linalg.norm(embedding2))
                    cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

                    # Tính Cosine Distance
                    distance = 1 - cosine_similarity
                    # print(distance)
                    if distance < 0.5:  # Ngưỡng nhận diện
                        x = face['facial_area']['x']
                        y = face['facial_area']['y']
                        w = face['facial_area']['w']
                        h = face['facial_area']['h']
                        detect.append([[x, y, w, h], face['confidence'], row["identity"]])
                        break
            #print('represent xong')
            # if len(detect)>0:
            #     print(f'phát hiện {len(detect)} khuôn mặt')
            # Cập nhật,gán ID băằng DeepSort
            tracks = tracker.update_tracks(detect, frame = frame)
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    # Lấy toạ độ, class_id để vẽ lên hình ảnh
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if blur_name == 'pixelation':
                        frame = pixelate(frame, (x1, y1, x2-x1, y2-y1), pixel_size=10)
                    elif blur_name == 'gaussian blur':
                        roi = frame[y1:y2, x1:x2]
                        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                        frame[y1:y2, x1:x2] = blurred_roi
            # for track in detect:
            #     # if track.is_confirmed():
            #     #     track_id = track.track_id
            #         # Lấy toạ độ, class_id để vẽ lên hình ảnh
            #         # ltrb = track.to_ltrb()
            #     # class_id = track.get_det_class()
            #     x, y, w, h= track[0]
            #     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     if blur_name == 'pixelation':
            #         frame = pixelate(frame, (x, y, w, h), pixel_size=10)
            #     elif blur_name == 'gaussian blur':
            #         roi = frame[y:y+h, x:x+w]
            #         blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            #         frame[y:y+h, x:x+w] = blurred_roi
            #print('rectangle xong')        
        except Exception as e:
            print(f"Error processing frame: {e}")

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
    print("Xử lý xong")
    cap.release()
    out.release()

def generate_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()