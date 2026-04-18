"""
train_liveness.py
Mục đích: Script này chạy ở ngoài, dùng để train mô hình liveness.
Cấu trúc thư mục dữ liệu cần có (bạn cần tự bỏ ảnh vào):
  data/liveness_data/real/   <--- Chứa ảnh khuôn mặt thật
  data/liveness_data/fake/   <--- Chứa ảnh khuôn mặt giả (màn hình, giấy in)

Chạy: python train_liveness.py
"""

import os
from liveness_module import LivenessDetector

REAL_DIR = "data/liveness_data/real"
FAKE_DIR = "data/liveness_data/fake"

def prepare_folders():
    if not os.path.exists(REAL_DIR):
        os.makedirs(REAL_DIR)
        print(f"Đã tạo thư mục '{REAL_DIR}'. Vui lòng bỏ ảnh MẶT THẬT vào đây.")
    if not os.path.exists(FAKE_DIR):
        os.makedirs(FAKE_DIR)
        print(f"Đã tạo thư mục '{FAKE_DIR}'. Vui lòng bỏ ảnh MẶT GIẢ vào đây.")

if __name__ == "__main__":
    prepare_folders()
    
    # Kiểm tra xem có ảnh không
    real_count = len(os.listdir(REAL_DIR)) if os.path.exists(REAL_DIR) else 0
    fake_count = len(os.listdir(FAKE_DIR)) if os.path.exists(FAKE_DIR) else 0
    
    print(f"[{real_count}] ảnh Real, [{fake_count}] ảnh Fake.")
    
    if real_count > 0 and fake_count > 0:
        print("Bắt đầu huấn luyện...")
        detector = LivenessDetector()
        success = detector.train(REAL_DIR, FAKE_DIR, verbose=True)
        if success:
            print("Chúc mừng! Đã tạo thành công data/liveness_model.pkl")
    else:
        print("\nCHÚ Ý: Bạn chưa có đủ ảnh Real và Fake.")
        print("Vui lòng copy ảnh khuôn mặt cắt chuẩn vào 2 thư mục trên rồi chạy lại file này.")
