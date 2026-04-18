"""
ui.py
Mục đích: Thiết kế giao diện đồ họa bằng thư viện tkinter và gắn kết với các chức năng.

"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import datetime
import cv2
from PIL import Image, ImageTk
from utils import lay_gio_hien_tai, lay_ngay_hien_tai, dinh_dang_hien_thi
import database
import face_module
import liveness_module

class App:
    """
    Class khởi tạo và quản lý toàn bộ giao diện tkinter của ứng dụng Máy Chấm Công.
    """
    def __init__(self, root: tk.Tk):
        """
        Khởi tạo cửa sổ chính và các tab (Chấm công, Quản lý NV, Báo cáo).
        
        Args:
            root (tk.Tk): Cửa sổ gốc của ứng dụng.
        """
        self.root = root
        self.root.title("Hệ Thống Phần Mềm Máy Chấm Công (Co AI)")
        self.root.geometry("850x750")
        # Tuỳ chỉnh font mặc định
        self.root.option_add("*Font", "{Segoe UI} 11")
        
        # Bắt sự kiện tắt Window để giải phóng Camera
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Cấu hình AI
        self.camera_active = False
        self.camera = None
        self.recognizer = face_module.load_mo_hinh()
        self.face_cascade = cv2.CascadeClassifier(face_module.FACE_CASCADE_PATH)
        self.liveness_detector = liveness_module.LivenessDetector()
        self.last_attendance = {}
        
        # Style cho giao diện (Theme và màu sắc nút)
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Tabs container
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Khởi tạo 3 tab
        self.tab_cham_cong = ttk.Frame(self.notebook)
        self.tab_nhan_vien = ttk.Frame(self.notebook)
        self.tab_bao_cao = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_cham_cong, text="🕒 Chấm Công")
        self.notebook.add(self.tab_nhan_vien, text="👥 Quản lý Nhân viên")
        self.notebook.add(self.tab_bao_cao, text="📊 Báo cáo")
        
        # Gọi hàm thiết kế từng Tab
        self._setup_tab_cham_cong()
        self._setup_tab_nhan_vien()
        self._setup_tab_bao_cao()
        
        # Bắt đầu cập nhật đồng hồ
        self.cap_nhat_dong_ho()

    def _setup_tab_cham_cong(self):
        """
        Thiết kế UI cho Tab 'Chấm Công'.
        Bao gồm: Đồng hồ thời gian thực, Ô nhập mã NV, Các nút Check-in / Check-out, Label trạng thái.
        """
        frame_top = tk.Frame(self.tab_cham_cong, bg="#f0f8ff", pady=20)
        frame_top.pack(fill="x")
        
        # Đồng hồ
        self.lbl_clock = tk.Label(frame_top, text="Ngày Giờ", font=("Segoe UI", 24, "bold"), fg="#004085", bg="#f0f8ff")
        self.lbl_clock.pack(pady=10)
        
        # Khung Camera dùng Canvas thay cho Label để render ảnh chuẩn hơn
        self.canvas_camera = tk.Canvas(frame_top, bg="black", width=500, height=350)
        self.canvas_camera.pack()
        
        btn_toggle_cam = tk.Button(frame_top, text="📸 Bật/Tắt Camera", bg="#17a2b8", fg="white", command=self.toggle_camera)
        btn_toggle_cam.pack(pady=5)
        
        frame_mid = tk.Frame(self.tab_cham_cong, pady=10)
        frame_mid.pack()
        
        # Form nhập liệu
        tk.Label(frame_mid, text="Mã nhân viên:", font=("Segoe UI", 13)).grid(row=0, column=0, padx=10, pady=10)
        self.ent_ma_nv_cham_cong = tk.Entry(frame_mid, font=("Segoe UI", 14), width=20)
        self.ent_ma_nv_cham_cong.grid(row=0, column=1, padx=10, pady=10)
        # focus cho entry
        self.ent_ma_nv_cham_cong.focus()
        
        tk.Button(frame_mid, text="🔍 Tìm", command=self.tim_nv_cham_cong, bg="#f8f9fa", width=8).grid(row=0, column=2, padx=10)
        
        # Thông tin NV hiển thị
        self.lbl_ten_nv_cc = tk.Label(frame_mid, text="Họ tên: ---", font=("Segoe UI", 12, "bold"))
        self.lbl_ten_nv_cc.grid(row=1, column=0, columnspan=3, pady=(10, 5), sticky="w", padx=10)
        
        self.lbl_phong_ban_cc = tk.Label(frame_mid, text="Phòng ban: ---", font=("Segoe UI", 12))
        self.lbl_phong_ban_cc.grid(row=2, column=0, columnspan=3, pady=5, sticky="w", padx=10)
        
        # Nút chức năng
        frame_btn = tk.Frame(self.tab_cham_cong)
        frame_btn.pack(pady=20)
        
        btn_checkin = tk.Button(frame_btn, text="✅ CHECK-IN", bg="#28a745", fg="white", font=("Segoe UI", 14, "bold"), width=15, command=self.xu_ly_check_in)
        btn_checkin.pack(side="left", padx=20)
        
        btn_checkout = tk.Button(frame_btn, text="🚪 CHECK-OUT", bg="#dc3545", fg="white", font=("Segoe UI", 14, "bold"), width=15, command=self.xu_ly_check_out)
        btn_checkout.pack(side="left", padx=20)
        
        # Trạng thái thông báo
        self.lbl_status = tk.Label(self.tab_cham_cong, text="Sẵn sàng chấm công.", font=("Segoe UI", 12, "italic"), fg="#6c757d")
        self.lbl_status.pack(pady=30)

    def _setup_tab_nhan_vien(self):
        """
        Thiết kế UI cho Tab 'Quản lý Nhân Viên'.
        Bao gồm form thêm nhân viên và danh sách hiển thị dữ liệu (Treeview).
        """
        frame_form = ttk.LabelFrame(self.tab_nhan_vien, text="Thêm nhân viên mới", padding=(10, 10))
        frame_form.pack(fill="x", padx=10, pady=10)
        
        tk.Label(frame_form, text="Mã NV:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.ent_ma_nv = tk.Entry(frame_form, width=15)
        self.ent_ma_nv.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(frame_form, text="Họ tên:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.ent_ten_nv = tk.Entry(frame_form, width=25)
        self.ent_ten_nv.grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(frame_form, text="Phòng ban:").grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.ent_phong_ban = tk.Entry(frame_form, width=20)
        self.ent_phong_ban.grid(row=0, column=5, padx=5, pady=5)
        
        tk.Button(frame_form, text="➕ Thêm", command=self.them_nv_moi, bg="#007bff", fg="white", width=10).grid(row=0, column=6, padx=15)

        # Quản lý AI (Mới)
        frame_ai = ttk.LabelFrame(self.tab_nhan_vien, text="Nhận Diện Khuôn Mặt (AI)", padding=(10, 10))
        frame_ai.pack(fill="x", padx=10, pady=5)
        
        tk.Button(frame_ai, text="📸 Lấy mẫu khuôn mặt", command=self.thu_thap_mau, width=20, bg="#ffc107").pack(side="left", padx=10)
        tk.Button(frame_ai, text="🧠 Huấn luyện mô hình", command=self.huan_luyen, width=20, bg="#28a745", fg="white").pack(side="left", padx=10)
        tk.Label(frame_ai, text="(Lưu ý: Chọn 1 nhân viên ở bảng dưới rồi bấm 'Lấy mẫu')").pack(side="left", padx=10)

        # Treeview danh sách nhân viên
        self.trv_nv = ttk.Treeview(self.tab_nhan_vien, columns=("ma", "ten", "phong"), show="headings", height=15)
        self.trv_nv.heading("ma", text="Mã NV")
        self.trv_nv.heading("ten", text="Họ tên")
        self.trv_nv.heading("phong", text="Phòng ban")
        
        self.trv_nv.column("ma", width=100, anchor="center")
        self.trv_nv.column("ten", width=300)
        self.trv_nv.column("phong", width=250)
        self.trv_nv.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Load data mặc định
        self.tai_du_lieu_nhan_vien()

    def _setup_tab_bao_cao(self):
        """
        Thiết kế UI cho Tab 'Báo cáo'.
        Hiển thị dữ liệu chấm công và chức năng xuất CSV.
        """
        frame_filter = ttk.LabelFrame(self.tab_bao_cao, text="Bộ lọc dữ liệu", padding=(10, 10))
        frame_filter.pack(fill="x", padx=10, pady=10)
        
        tk.Label(frame_filter, text="Mã NV:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_bc_ma = tk.Entry(frame_filter, width=12)
        self.ent_bc_ma.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(frame_filter, text="Từ ngày (YYYY-MM-DD):").grid(row=0, column=2, padx=5, pady=5)
        self.ent_bc_tu = tk.Entry(frame_filter, width=12)
        self.ent_bc_tu.insert(0, lay_ngay_hien_tai())
        self.ent_bc_tu.grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(frame_filter, text="Đến ngày (YYYY-MM-DD):").grid(row=0, column=4, padx=5, pady=5)
        self.ent_bc_den = tk.Entry(frame_filter, width=12)
        self.ent_bc_den.insert(0, lay_ngay_hien_tai())
        self.ent_bc_den.grid(row=0, column=5, padx=5, pady=5)
        
        tk.Button(frame_filter, text="🔍 Lọc", command=self.loc_bao_cao, bg="#17a2b8", fg="white", width=8).grid(row=0, column=6, padx=10)
        tk.Button(frame_filter, text="📥 Xuất CSV", command=self.xuat_bao_cao_csv, bg="#ffc107", fg="black", width=12).grid(row=0, column=7, padx=10)

        # Treeview báo cáo
        self.trv_bc = ttk.Treeview(self.tab_bao_cao, columns=("ma", "ten", "ngay", "vao", "ra", "tong"), show="headings", height=15)
        for col, title in zip(("ma", "ten", "ngay", "vao", "ra", "tong"), ("Mã NV", "Họ Tên", "Ngày", "Giờ Vào", "Giờ Ra", "Tổng Giờ")):
            self.trv_bc.heading(col, text=title)
            self.trv_bc.column(col, width=120, anchor="center")
            
        self.trv_bc.column("ten", width=200, anchor="w")
        self.trv_bc.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Thống kê tổng kết
        self.lbl_thong_ke = tk.Label(self.tab_bao_cao, text="Tổng số bản ghi: 0 | Tổng giờ làm việc: 0 giờ", font=("Segoe UI", 11, "bold"))
        self.lbl_thong_ke.pack(pady=10, anchor="w", padx=10)

        # Load data mặc định
        self.loc_bao_cao()

    def cap_nhat_dong_ho(self):
        """
        Hàm đệ quy cập nhật thời gian thực trên giao diện bằng self.root.after().
        Được gọi lại mỗi giây (1000ms), không gây treo GUI.
        """
        now = datetime.datetime.now()
        # Định dạng thứ, ngày
        thu = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        ngay_chuoi = f"{thu[now.weekday()]}, {now.strftime('%d/%m/%Y')} — {now.strftime('%H:%M:%S')}"
        
        self.lbl_clock.config(text=ngay_chuoi)
        self.root.after(1000, self.cap_nhat_dong_ho)
        
    def thong_bao_trang_thai(self, noi_dung: str, is_error: bool = False):
        """
        Hiển thị câu trạng thái (status).
        
        Args:
            noi_dung (str): Nội dung cần hiển thị.
            is_error (bool): Tuỳ chọn màu chữ. Nếu lỗi thì màu đỏ, otherwise màu xanh lục.
        """
        color = "red" if is_error else "green"
        self.lbl_status.config(text=noi_dung, fg=color)
        
    def tim_nv_cham_cong(self):
        """
        Tìm kiếm thông tin nhân viên theo mã nhập vào ở tab Chấm Công.
        Hiển thị tên và phòng ban nếu tìm thấy, thông báo lỗi nếu không.
        """
        ma_nv = self.ent_ma_nv_cham_cong.get().strip().upper()
        if not ma_nv:
            self.thong_bao_trang_thai("Vui lòng nhập mã nhân viên để tìm!", True)
            return
            
        nv = database.tim_nhan_vien(ma_nv)
        if nv:
            self.lbl_ten_nv_cc.config(text=f"Họ tên: {nv[2]}", fg="black")
            self.lbl_phong_ban_cc.config(text=f"Phòng ban: {nv[3]}", fg="black")
            self.thong_bao_trang_thai("Đã tìm thấy nhân viên.", False)
        else:
            self.lbl_ten_nv_cc.config(text="Họ tên: ---", fg="black")
            self.lbl_phong_ban_cc.config(text="Phòng ban: ---", fg="black")
            self.thong_bao_trang_thai("Không tìm thấy nhân viên hợp lệ!", True)

    def xu_ly_check_in(self):
        """
        Sự kiện nhấn nút Check-in: Gọi hàm database check_in và thông báo kết quả.
        Đồng thời tự động tìm/hiển thị tên nếu chưa kịp ấn "Tìm".
        """
        ma_nv = self.ent_ma_nv_cham_cong.get().strip().upper()
        if not ma_nv:
            self.thong_bao_trang_thai("Vui lòng nhập mã nhân viên!", True)
            return
            
        # Hiển thị tên (nếu có)
        self.tim_nv_cham_cong()
        
        ket_qua = database.check_in(ma_nv)
        is_loi = not ket_qua["thanh_cong"]
        self.thong_bao_trang_thai(ket_qua["thong_bao"], is_loi)
        
        if not is_loi:
            self.loc_bao_cao() # Cập nhật view

    def xu_ly_check_out(self):
        """
        Sự kiện nhấn nút Check-out: Gọi hàm database check_out và thông báo kết quả.
        """
        ma_nv = self.ent_ma_nv_cham_cong.get().strip().upper()
        if not ma_nv:
            self.thong_bao_trang_thai("Vui lòng nhập mã nhân viên!", True)
            return
            
        self.tim_nv_cham_cong()
        
        ket_qua = database.check_out(ma_nv)
        is_loi = not ket_qua["thanh_cong"]
        self.thong_bao_trang_thai(ket_qua["thong_bao"], is_loi)
        
        if not is_loi:
            self.loc_bao_cao()

    def tu_dong_cham_cong(self, ma_nv):
        """
        Tự động chấm công với thời gian cooldown 5 giây.
        """
        hien_tai = time.time()
        lan_cuoi = self.last_attendance.get(ma_nv, 0)
        if hien_tai - lan_cuoi < 5:
            return # Chưa qua 5 giây
        
        # Thử Check-in trước
        kq_in = database.check_in(ma_nv)
        if kq_in["thanh_cong"]:
            self.thong_bao_trang_thai(f"[TỰ ĐỘNG] {kq_in['thong_bao']}", False)
        else:
            # Nghĩa là đã check-in, thử Check-out (cập nhật giờ ra)
            kq_out = database.check_out(ma_nv)
            if kq_out["thanh_cong"]:
                self.thong_bao_trang_thai(f"[TỰ ĐỘNG] {kq_out['thong_bao']}", False)
        
        self.last_attendance[ma_nv] = hien_tai
        self.loc_bao_cao() # Cập nhật bảng báo cáo

    def tai_du_lieu_nhan_vien(self):
        """
        Load toàn bộ dữ liệu nhân viên từ CSDL lên Treeview.
        """
        for item in self.trv_nv.get_children():
            self.trv_nv.delete(item)
            
        rows = database.lay_tat_ca_nhan_vien()
        for row in rows:
            self.trv_nv.insert("", "end", values=(row[1], row[2], row[3]))

    def them_nv_moi(self):
        """
        Sự kiện nút 'Thêm' trong tab Quản lý nhân viên.
        """
        ma = self.ent_ma_nv.get().strip().upper()
        ten = self.ent_ten_nv.get().strip()
        phong = self.ent_phong_ban.get().strip()
        
        if not ma or not ten:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ Mã NV và Họ Tên!")
            return
            
        kq = database.them_nhan_vien(ma, ten, phong)
        if kq["thanh_cong"]:
            messagebox.showinfo("Thành công", kq["thong_bao"])
            self.ent_ma_nv.delete(0, tk.END)
            self.ent_ten_nv.delete(0, tk.END)
            self.ent_phong_ban.delete(0, tk.END)
            self.tai_du_lieu_nhan_vien()
        else:
            messagebox.showerror("Lỗi", kq["thong_bao"])

    def loc_bao_cao(self):
        """
        Lấy thông tin lọc (mã nv, từ ngày, đến ngày) từ UI và tải lên Treeview báo cáo.
        Cập nhật cả dòng Label thống kê tổng quát.
        """
        ma = self.ent_bc_ma.get().strip().upper()
        tu = self.ent_bc_tu.get().strip()
        den = self.ent_bc_den.get().strip()
        
        # Xóa dữ liệu cũ
        for item in self.trv_bc.get_children():
            self.trv_bc.delete(item)
            
        # Lấy dữ liệu
        try:
            rows = database.lay_cham_cong_tu_den(tu, den, ma if ma else None)
            
            tong_tat_ca_gio = 0.0
            for r in rows:
                tong_gio_hien_thi = dinh_dang_hien_thi(r[5] if r[5] else 0.0)
                # Tính tổng
                if r[5]:
                    tong_tat_ca_gio += r[5]
                self.trv_bc.insert("", "end", values=(r[0], r[1], r[2], r[3] or "", r[4] or "", tong_gio_hien_thi))
                
            self.lbl_thong_ke.config(text=f"Tổng số bản ghi: {len(rows)} | Tổng giờ làm việc của danh sách lọc: {round(tong_tat_ca_gio, 2)} giờ")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi truy vấn báo cáo: {str(e)}")

    def xuat_bao_cao_csv(self):
        """
        Sự kiện nhấn nút Xuất CSV. Ghi dữ liệu hiện tại trong Treeview ra file "bao_cao.csv".
        """
        # Thu thập dữ liệu từ treeview
        ds = []
        for iid in self.trv_bc.get_children():
            ds.append(self.trv_bc.item(iid)['values'])
            
        if not ds:
            messagebox.showinfo("Thông báo", "Không có dữ liệu để xuất.")
            return
            
        file_name = f"bao_cao_cham_cong_{lay_ngay_hien_tai()}.csv"
        kq = database.xuat_csv(ds, file_name)
        if kq["thanh_cong"]:
            messagebox.showinfo("Thành công", kq["thong_bao"])
        else:
            messagebox.showerror("Lỗi", kq["thong_bao"])

    # -------------- CÁC HÀM XỬ LÝ AI -----------------

    def thu_thap_mau(self):
        selected = self.trv_nv.selection()
        if not selected:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn 1 nhân viên từ danh sách trước!")
            return
            
        item = self.trv_nv.item(selected[0])
        ma_nv = item['values'][0]
        
        # Tắt camera chấm công tạm thời để tránh xung đột
        was_active = self.camera_active
        if self.camera_active:
            self.toggle_camera()
            
        kq = face_module.thu_thap_khuon_mat(ma_nv)
        if kq["thanh_cong"]:
            messagebox.showinfo("Thành công", kq["thong_bao"])
        else:
            messagebox.showerror("Lỗi", kq["thong_bao"])
            
        # Bật lại nếu nó từng mở
        if was_active:
            self.toggle_camera()

    def huan_luyen(self):
        self.thong_bao_trang_thai("Đang huấn luyện AI...", False)
        self.root.update()
        
        kq = face_module.huan_luyen_mo_hinh()
        if kq["thanh_cong"]:
            messagebox.showinfo("Thành công", kq["thong_bao"])
            self.recognizer = face_module.load_mo_hinh()
            self.thong_bao_trang_thai("Đã huấn luyện xong.", False)
        else:
            messagebox.showerror("Lỗi", kq["thong_bao"])
            self.thong_bao_trang_thai("Lỗi huấn luyện.", True)
            
    def toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            # Dùng cv2.CAP_DSHOW trên Windows giúp mở camera nhanh và ổn định hơn
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.thong_bao_trang_thai("Không tìm thấy Camera hệ thống!", True)
                self.camera_active = False
                return
            self.thong_bao_trang_thai("Đang bật Camera...", False)
            self.cap_nhat_camera()
        else:
            if self.camera:
                self.camera.release()
                self.camera = None
            self.canvas_camera.delete("all")
            self.thong_bao_trang_thai("Camera đã tắt.", False)
            
    def cap_nhat_camera(self):
        if not self.camera_active or not self.camera:
            return
            
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_resized = cv2.resize(frame, (500, 350)) # resize cho vừa khung Label 500x350
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # Vẽ khung chữ nhật nhận diện mặt bất kể AI đã load hay chưa
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Kiểm tra liveness bằng LBP và phân tích phản xạ ánh sáng
                is_real_lbp = self.liveness_detector.predict(face_roi)
                is_real_spec, spec_score, spec_info = liveness_module.analyze_specular_reflection(frame_resized, (x, y, w, h))
                is_real = is_real_lbp and is_real_spec

                if is_real:
                    cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Nhận diện nếu mô hình đã load
                    if self.recognizer:
                        id, confidence = self.recognizer.predict(face_roi)
                        
                        if confidence < 75:
                            ma_nv = database.lay_ma_nv_tu_id(id)
                            if ma_nv:
                                # Điền mã NV vào ô text và chạy tìm
                                if self.ent_ma_nv_cham_cong.get() != ma_nv:
                                    self.ent_ma_nv_cham_cong.delete(0, tk.END)
                                    self.ent_ma_nv_cham_cong.insert(0, ma_nv)
                                    self.tim_nv_cham_cong()
                                    
                                self.tu_dong_cham_cong(ma_nv)
                                cv2.putText(frame_resized, str(ma_nv), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame_resized, "Khong the xac dinh", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            self.lbl_ten_nv_cc.config(text="LỖI: KHÔNG XÁC ĐỊNH", fg="orange")
                            self.lbl_phong_ban_cc.config(text="Người lạ trong khung hình", fg="orange")
                            if self.ent_ma_nv_cham_cong.get() != "":
                                self.ent_ma_nv_cham_cong.delete(0, tk.END)
                    else:
                        cv2.putText(frame_resized, "Chua huan luyen AI", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # Phát hiện giả mạo
                    cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame_resized, "GIA MAO (FAKE)", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self.lbl_ten_nv_cc.config(text="CẢNH BÁO: PHÁT HIỆN GIẢ MẠO", fg="red")
                    self.lbl_phong_ban_cc.config(text="Vui lòng dùng mặt thật", fg="red")
                    if self.ent_ma_nv_cham_cong.get() != "":
                        self.ent_ma_nv_cham_cong.delete(0, tk.END)
            
            # Chuyển frame thành Ảnh cho Tkinter
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Cập nhật ảnh lên Canvas thay vì Label
            self.canvas_camera.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas_camera.imgtk = imgtk
            
        self.root.after(30, self.cap_nhat_camera)

    def on_closing(self):
        if self.camera:
            self.camera.release()
        self.root.destroy()
