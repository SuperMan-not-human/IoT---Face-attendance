"""
face_module.py
Mục đích: Xử lý các tác vụ nhận diện khuôn mặt bằng OpenCV.
Bao gồm: Thu thập ảnh mẫu, huấn luyện mô hình (LBPH), và nhận diện trên webcam.

"""

import cv2
import os
import numpy as np
from PIL import Image
import database

# Đường dẫn cần thiết
FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
DATA_FACES_DIR = "data/faces"
TRAINER_FILE = "data/trainer.yml"

# Đảm bảo thư mục tồn tại
if not os.path.exists(DATA_FACES_DIR):
    os.makedirs(DATA_FACES_DIR)

def thu_thap_khuon_mat(ma_nv: str):
    """
    Bật webcam và thu thập 30 bức ảnh cắt khuôn mặt của nhân viên.
    
    Args:
        ma_nv (str): Mã nhân viên cần lấy mẫu
        
    Returns:
        dict: Kết quả {'thanh_cong': bool, 'thong_bao': str}
    """
    # Lấy ID số nguyên của nhân viên từ DB vì LBPH chỉ nhận interger ID
    nv = database.tim_nhan_vien(ma_nv)
    if not nv:
        return {"thanh_cong": False, "thong_bao": "Không tìm thấy nhân viên."}
    
    nv_id = nv[0] # ID số nguyên hệ thống
    
    # Sử dụng cv2.CAP_DSHOW để camera trên Windows luôn mở nhanh chóng và ko bị màn hình đen
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        return {"thanh_cong": False, "thong_bao": "Không thể mở WebCam. Vui lòng kiểm tra quyền hoặc thiết bị."}
        
    cam.set(3, 640) # Chiều rộng
    cam.set(4, 480) # Chiều cao

    face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    count = 0
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                break
                
            img = cv2.flip(img, 1) # Lật ảnh lại cho đỡ ngược
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)     
                count += 1
                # Luu anh cat khuon mat vao thu muc data/faces
                # Dinh dang ten file: User.id.count.jpg
                file_path = f"{DATA_FACES_DIR}/User.{nv_id}.{count}.jpg"
                face_roi = gray[y:y+h, x:x+w]
                # Sử dụng CLAHE để cân bằng sáng chống ngược sáng/thiếu sáng
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                face_roi = clahe.apply(face_roi)
                face_roi = cv2.resize(face_roi, (100, 100))
                cv2.imwrite(file_path, face_roi)

                cv2.putText(img, f"Dang chup: {count}/30", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Thu thap hinh anh [Nhan ESC hoac doi 30 giay]', img)

            k = cv2.waitKey(100) & 0xff # Chờ 100ms
            if k == 27: # Bấm ESC để thoát sớm
                break
            elif count >= 30: # Lấy đủ 30 ảnh mẫu thì thoát
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        
    if count == 0:
        return {"thanh_cong": False, "thong_bao": "Không tìm thấy khuôn mặt nào. Hãy thử lại."}
        
    return {"thanh_cong": True, "thong_bao": f"Thu thập thành công {count} ảnh khuôn mặt cho nhân viên {ma_nv}."}


def huan_luyen_mo_hinh():
    """
    Đọc tất cả ảnh trong thư mục data/faces và huấn luyện mô hình bằng LBPH.
    Lưu kết quả ra file data/trainer.yml.
    
    Returns:
        dict: Kết quả {'thanh_cong': bool, 'thong_bao': str}
    """
    image_paths = [os.path.join(DATA_FACES_DIR, f) for f in os.listdir(DATA_FACES_DIR) if f.startswith("User.")]
    if not image_paths:
        return {"thanh_cong": False, "thong_bao": "Không có dữ liệu ảnh nào trong mục data/faces."}
        
    face_samples = []
    ids = []
    
    # Load detector
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    for image_path in image_paths:
        # Mo anh bang Pillow va chuyen sang thang do xam
        try:
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Lay ID tu ten file `User.{id}.{count}.jpg`
            id = int(os.path.split(image_path)[-1].split(".")[1])

            # Ảnh thu thập đã được xử lý cân bằng sáng CLAHE và resize sẵn
            # Phòng trường hợp ảnh cũ chưa resize, ta check và resize
            if img_numpy.shape != (100, 100):
                img_numpy = cv2.resize(img_numpy, (100, 100))

            face_samples.append(img_numpy)
            ids.append(id)

        except Exception as e:
            print(f"Loi khi doc file {image_path}: {e}")

    if not face_samples:
        return {"thanh_cong": False, "thong_bao": "Không trích xuất được khuôn mặt hợp lệ từ các ảnh."}

    # Khởi tạo thuật toán LBPH
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        # Nếu thư viện thiếu `contrib`
        return {"thanh_cong": False, "thong_bao": "Vui lòng cài đặt: pip install opencv-contrib-python"}
        
    try:
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(TRAINER_FILE) 
        return {"thanh_cong": True, "thong_bao": f"Huấn luyện xong {len(np.unique(ids))} nhân viên."}
    except Exception as e:
        return {"thanh_cong": False, "thong_bao": f"Lỗi huấn luyện: {str(e)}"}


def load_mo_hinh():
    """
    Tải mô hình đã huấn luyện từ file trainer.yml.
    """
    if not os.path.exists(TRAINER_FILE):
        return None
        
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)
        return recognizer
    except:
        return None
