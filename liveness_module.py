"""
liveness_module.py
Mục đích: Module kiểm tra liveness (giống/ảnh thực vs giả mạo) sử dụng thuật toán LBP.

"""

import cv2
import numpy as np
import os
import pickle

# Nếu thiếu thư viện, cần pip install scikit-image scikit-learn
try:
    from skimage import feature
    from sklearn.svm import SVC
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

MODEL_PATH = "data/liveness_model.pkl"

class LivenessDetector:
    def __init__(self):
        self.model = None
        self.radius = 3
        self.n_points = 24
        self.load_model()
        
    def extract_lbp(self, image):
        """
        Trích xuất đặc trưng LBP từ ảnh.
        Yêu cầu đầu vào: ảnh xám (grayscale).
        """
        if not _SKIMAGE_AVAILABLE:
            # Fallback nếu không có skimage, nhưng LBP cần skimage để chuẩn nhất
            return None
            
        lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))
        
        # Chuẩn hoá histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def train(self, real_faces_dir, fake_faces_dir, verbose=False):
        """
        Huấn luyện mô hình Liveness SVM bằng đặc trưng LBP.
        """
        if not _SKIMAGE_AVAILABLE:
            if verbose: print("Lỗi: Cần cài đặt scikit-image và scikit-learn (pip install scikit-image scikit-learn)")
            return False

        data = []
        labels = []
        
        # Đọc ảnh Real (nhãn 1)
        if os.path.exists(real_faces_dir):
            for f in os.listdir(real_faces_dir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    path = os.path.join(real_faces_dir, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        hist = self.extract_lbp(img)
                        if hist is not None:
                            data.append(hist)
                            labels.append(1)
        
        # Đọc ảnh Fake (nhãn 0)
        if os.path.exists(fake_faces_dir):
            for f in os.listdir(fake_faces_dir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    path = os.path.join(fake_faces_dir, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        hist = self.extract_lbp(img)
                        if hist is not None:
                            data.append(hist)
                            labels.append(0)

        if len(data) == 0:
            if verbose: print("Không có dữ liệu huấn luyện!")
            return False
            
        if verbose: print(f"Đang huấn luyện với {len(data)} mẫu...")
        
        model = SVC(kernel="linear", probability=True)
        model.fit(data, labels)
        
        # Lưu mô hình
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
            
        self.model = model
        if verbose: print("Đã huấn luyện và lưu liveness mô hình thành công.")
        return True

    def load_model(self):
        """
        Tải mô hình Liveness (nếu có).
        """
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"Lỗi khi tải Liveness Model: {e}")
                self.model = None

    def predict(self, face_img):
        """
        Dự đoán ảnh face là Real hay Fake.
        face_img: Ảnh xám khuôn mặt (Grayscale crop vùng có mặt).
        Trả về: True (Real), False (Fake).
        Nếu chưa có mô hình hoặc chưa có thư viện, luôn trả về True (bỏ qua check).
        """
        # Nếu không có sklearn/skimage hoặc không có model đã train, skip pass cho Real
        if not _SKIMAGE_AVAILABLE or self.model is None:
            return True
            
        # Resize về cỡ chuẩn
        face_img = cv2.resize(face_img, (100, 100))
        hist = self.extract_lbp(face_img)
        
        if hist is None:
            return True # Lỗi trích xuất
            
        prediction = self.model.predict([hist])[0]
        prob = self.model.predict_proba([hist])[0]
        
        # Hạ ngưỡng confidence từ 0.6 xuống 0.5 hoặc chỉ cần check prediction để tránh nhận sai mặt thật (false negative)
        if prediction == 1 and prob[1] >= 0.5: 
            return True # Real
        else:
            return False # Fake

def analyze_specular_reflection(frame, bbox):
    """
    Phân tích chống giả mạo bằng cách phân tích phản xạ ánh sáng (Specular Analysis).
    
    Args:
        frame: Ảnh toàn bộ camera (numpy array BGR)
        bbox: tuple (x, y, w, h) tọa độ khuôn mặt
    
    Returns:
        (is_real (bool), confidence_score (float), info_dict (dict))
    """
    x, y, w, h = bbox
    # Trích xuất vùng khuôn mặt
    face_roi = frame[y:y+h, x:x+w]
    
    if face_roi.size == 0:
        return False, 0.0, {"error": "Invalid bbox"}

    # 1. Tách kênh Luminance: Chuyển sang YCrCb và lấy Y (Độ sáng)
    ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    
    # 2. Ngưỡng thích nghi (Adaptive Threshold)
    mean_Y = np.mean(Y)
    std_Y = np.std(Y) + 1e-6
    
    # Đặt mốc vùng cháy sáng là sáng hơn trung bình 35% (hệ số 1.35)
    # Ràng buộc ngưỡng không vượt quá 250 để vẫn có thể bắt điểm cực sáng
    threshold = min(250, mean_Y * 1.35)
    
    # 3. Phát hiện vùng cháy sáng (Hotspots)
    _, mask = cv2.threshold(Y, threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    # Tính diện tích
    total_area = w * h
    hotspot_area = cv2.countNonZero(mask)
    area_ratio = hotspot_area / total_area
    
    # Điểm phạt bắt đầu từ 1.0 (Real) -> Trừ dần nếu có dấu hiệu Fake
    score = 1.0
    info = {"area_ratio": area_ratio}

    # Logic: Diện tích vùng lóa > 15% -> Khả năng cực cao là màn hình đang chói sáng
    if area_ratio > 0.15:
        score -= 0.6
        info["reason"] = "Quá nhiều vùng lóa (Màn hình)"
    
    # 4. Phân tích cấu trúc vùng sáng (Reflection Geometry - Gradient)
    if hotspot_area > 0:
        # Tính gradient bằng Sobel (Vectorized)
        grad_x = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        
        # Tìm viền của các vùng lóa bằng Dilation - Original Mask
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(mask, kernel, iterations=1) - mask
        
        # Lấy độ sắc nét trung bình tại viền đó
        edge_pixels = grad_mag[edges > 0]
        if len(edge_pixels) > 0:
            mean_edge_grad = np.mean(edge_pixels)
            info["edge_gradient"] = mean_edge_grad
            
            # Cạnh càng sắc nét (gradient cao) -> Càng giống hình học của bóng màn hình
            if mean_edge_grad > 30: 
                score -= 0.2
                info["reason_grad"] = "Viền lóa quá sắc nét (Màn hình/Giấy)"
    
    # 5. Phân tích Histogram (Độ lệch Skewness)
    # Skewness = E[(Y - mean)^3] / std^3
    skewness = np.mean(((Y - mean_Y) / std_Y)**3)
    info["skewness"] = skewness
    
    # Nếu skewness âm quá mức (Tập trung hết vào phần sáng, cụt highlight)
    if skewness < -0.3:
        score -= 0.1
        info["reason_skew"] = "Cụt sáng Histogram (Màn hình)"

    # Đảm bảo score trong [0, 1]
    score = max(0.0, min(1.0, score))
    
    # Đánh giá cuối cùng: Điểm > 0.5 là Thật
    is_real = bool(score >= 0.6)
    
    return is_real, score, info
