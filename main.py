"""
main.py
Mục đích: Khởi chạy ứng dụng máy chấm công, khởi tạo CSDL lần đầu.

"""

import tkinter as tk
import database
from ui import App

def them_du_lieu_mau_neu_trong():
    """
    Hàm thêm 5 nhân viên vào csdl mẫu nếu dữ liệu trống để test.
    Chỉ chạy khi chưa có nhân viên nào trong bảng.
    """
    nhan_viens = database.lay_tat_ca_nhan_vien()
    if not nhan_viens:
        # Nếu danh sách rỗng, thêm dữ liệu mẫu
        du_lieu_mau = [
            ("NV001", "Nguyễn Văn An",   "Kỹ thuật"),
            ("NV002", "Trần Thị Bình",   "Kế toán"),
            ("NV003", "Lê Hoàng Cường",  "Kinh doanh"),
            ("NV004", "Phạm Thị Dung",   "Nhân sự"),
            ("NV005", "Hoàng Minh Đức",  "IT"),
        ]
        
        for nv in du_lieu_mau:
            database.them_nhan_vien(nv[0], nv[1], nv[2])

def main():
    """
    Hàm main()
    1. Khởi tạo cấu trúc các bảng CSDL.
    2. Thêm dữ liệu mẫu vào Bảng NhanVien.
    3. Tạo cửa sổ Tkinter và chạy vòng lặp sự kiện chính.
    """
    # 1. Tạo bảng (nếu chưa có)
    database.tao_bang()
    
    # Thêm dữ liệu mẫu
    them_du_lieu_mau_neu_trong()

    # 3. Khởi tạo App
    root = tk.Tk()
    app = App(root)
    
    # 4. Chạy event loop
    root.mainloop()

if __name__ == "__main__":
    main()
