"""
database.py
Mục đích: Xử lý toàn bộ thao tác với CSDL SQLite (kết nối, tạo bảng, thêm/sửa/xoá/truy vấn).

"""

import sqlite3
import csv
from utils import lay_gio_hien_tai, lay_ngay_hien_tai, tinh_so_gio

DB_NAME = "cham_cong.db"

def ket_noi_db() -> sqlite3.Connection:
    """
    Tạo hoặc kết nối tới file database SQLite.
    
    Returns:
        sqlite3.Connection: Đối tượng kết nối đến CSDL.
    """
    return sqlite3.connect(DB_NAME)

def tao_bang():
    """
    Tạo cấu trúc các bảng: `nhan_vien` và `cham_cong` nếu chưa tồn tại.
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    
    # Bảng nhân viên
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nhan_vien (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ma_nv TEXT UNIQUE NOT NULL,
            ten_nv TEXT NOT NULL,
            phong_ban TEXT
        )
    """)
    
    # Bảng chấm công
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cham_cong (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ma_nv TEXT NOT NULL,
            ngay TEXT NOT NULL,
            gio_vao TEXT,
            gio_ra TEXT,
            tong_gio REAL,
            FOREIGN KEY (ma_nv) REFERENCES nhan_vien (ma_nv)
        )
    """)
    
    conn.commit()
    conn.close()

def lay_tat_ca_nhan_vien() -> list:
    """
    Lấy danh sách toàn bộ nhân viên trong CSDL.
    
    Returns:
        list: Danh sách các tuple dòng dữ liệu nhân viên (id, ma_nv, ten_nv, phong_ban).
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, ma_nv, ten_nv, phong_ban FROM nhan_vien ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def them_nhan_vien(ma_nv: str, ten_nv: str, phong_ban: str) -> dict:
    """
    Thêm mới nhân viên vào thư viện CSDL.
    
    Args:
        ma_nv (str): Mã nhân viên (duy nhất).
        ten_nv (str): Họ tên đầy đủ.
        phong_ban (str): Tên phòng làm việc.
        
    Returns:
        dict: {"thanh_cong": bool, "thong_bao": str}
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    
    # Kiểm tra mã nhân viên trùng
    cursor.execute("SELECT id FROM nhan_vien WHERE ma_nv = ?", (ma_nv,))
    if cursor.fetchone():
        conn.close()
        return {"thanh_cong": False, "thong_bao": "Mã nhân viên đã tồn tại!"}
        
    try:
        cursor.execute(
            "INSERT INTO nhan_vien (ma_nv, ten_nv, phong_ban) VALUES (?, ?, ?)",
            (ma_nv, ten_nv, phong_ban)
        )
        conn.commit()
        thanh_cong = True
        thong_bao = "Thêm nhân viên thành công."
    except Exception as e:
        thanh_cong = False
        thong_bao = f"Lỗi hệ thống: {str(e)}"
    finally:
        conn.close()
        
    return {"thanh_cong": thanh_cong, "thong_bao": thong_bao}

def tim_nhan_vien(ma_nv: str) -> tuple:
    """
    Tìm thông tin nhân viên theo mã.
    
    Args:
        ma_nv (str): Mã nhân viên cần tìm.
        
    Returns:
        tuple: (id, ma_nv, ten_nv, phong_ban) nếu tìm thấy, ngược lại trả về None.
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, ma_nv, ten_nv, phong_ban FROM nhan_vien WHERE ma_nv = ?", (ma_nv,))
    row = cursor.fetchone()
    conn.close()
    return row

def check_in(ma_nv: str) -> dict:
    """
    Ghi nhận giờ vào làm cho nhân viên hôm nay.
    
    Kiểm tra nhân viên có tồn tại hay không. Nếu không, báo lỗi.
    Nếu có, kiểm tra đã check-in hôm nay chưa.
    Nếu chưa, thì vào tạo bản ghi giờ hiện tại.
    Nếu rồi, báo lỗi.

    Args:
        ma_nv (str): Mã nhân viên (ví dụ: "NV001").

    Returns:
        dict: Kết quả {'thanh_cong': bool, 'thong_bao': str}
    """
    # Không tìm thấy nhân viên
    nv = tim_nhan_vien(ma_nv)
    if not nv:
        return {"thanh_cong": False, "thong_bao": "Không tìm thấy nhân viên."}
        
    conn = ket_noi_db()
    cursor = conn.cursor()
    ngay_hom_nay = lay_ngay_hien_tai()

    # Truy vấn xem đã có bản ghi check-in hôm nay chưa
    cursor.execute(
        "SELECT id, gio_vao FROM cham_cong WHERE ma_nv = ? AND ngay = ?",
        (ma_nv, ngay_hom_nay)
    )
    ban_ghi_hien_co = cursor.fetchone()

    # Nếu đã check-in rồi → không cho check-in lại
    if ban_ghi_hien_co and ban_ghi_hien_co[1]:
        gio_vao_da_co = ban_ghi_hien_co[1]
        conn.close()
        return {"thanh_cong": False, "thong_bao": f"Đã check-in hôm nay lúc {gio_vao_da_co}"}

    # Chưa check-in → ghi bản ghi mới
    gio_vao = lay_gio_hien_tai()
    cursor.execute(
        "INSERT INTO cham_cong (ma_nv, ngay, gio_vao) VALUES (?, ?, ?)",
        (ma_nv, ngay_hom_nay, gio_vao)
    )
    conn.commit()
    conn.close()

    return {"thanh_cong": True, "thong_bao": f"Check-in thành công lúc {gio_vao}"}

def check_out(ma_nv: str) -> dict:
    """
    Ghi nhận giờ ra về cho nhân viên và tính tổng giờ làm.
    
    Yêu cầu: Nhân viên phải check-in hôm nay trước khi có thể check-out.
    Nếu chưa, báo lỗi. Nếu đã check-out, có thể cập nhật lại giờ check-out mới.

    Args:
        ma_nv (str): Mã nhân viên.

    Returns:
        dict: Kết quả chứa 'thanh_cong', 'thong_bao'.
    """
    nv = tim_nhan_vien(ma_nv)
    if not nv:
        return {"thanh_cong": False, "thong_bao": "Không tìm thấy nhân viên."}
        
    conn = ket_noi_db()
    cursor = conn.cursor()
    ngay_hom_nay = lay_ngay_hien_tai()

    # Truy vấn lấy bản ghi hôm nay
    cursor.execute(
        "SELECT id, gio_vao FROM cham_cong WHERE ma_nv = ? AND ngay = ?",
        (ma_nv, ngay_hom_nay)
    )
    ban_ghi = cursor.fetchone()

    if not ban_ghi or not ban_ghi[1]:
        conn.close()
        return {"thanh_cong": False, "thong_bao": "Chưa có dữ liệu check-in hôm nay"}

    ban_ghi_id = ban_ghi[0]
    gio_vao = ban_ghi[1]
    
    # Tính số giờ
    gio_ra = lay_gio_hien_tai()
    tong_gio = tinh_so_gio(gio_vao, gio_ra)

    # Cập nhật thông tin check-out
    cursor.execute(
        "UPDATE cham_cong SET gio_ra = ?, tong_gio = ? WHERE id = ?",
        (gio_ra, tong_gio, ban_ghi_id)
    )
    conn.commit()
    conn.close()

    return {"thanh_cong": True, "thong_bao": f"Check-out thành công lúc {gio_ra}"}

def lay_cham_cong_theo_ngay(ngay: str) -> list:
    """
    Trả về danh sách chấm công vào một ngày cụ thể (có kèm tên nhân viên).
    
    Args:
        ngay (str): Ngày cần truy vấn định dạng "YYYY-MM-DD".
        
    Returns:
        list: Các tuple (ma_nv, ten_nv, ngay, gio_vao, gio_ra, tong_gio).
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    query = """
        SELECT c.ma_nv, n.ten_nv, c.ngay, c.gio_vao, c.gio_ra, c.tong_gio
        FROM cham_cong c
        JOIN nhan_vien n ON c.ma_nv = n.ma_nv
        WHERE c.ngay = ?
        ORDER BY c.id DESC
    """
    cursor.execute(query, (ngay,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def lay_cham_cong_tu_den(tu_ngay: str, den_ngay: str, ma_nv: str = None) -> list:
    """
    Lấy danh sách chấm công trong khoảng thời gian, lọc tùy chọn theo nhân viên.
    
    Args:
        tu_ngay (str): Từ ngày dạng "YYYY-MM-DD"
        den_ngay (str): Đến ngày dạng "YYYY-MM-DD"
        ma_nv (str, optional): Nếu có, chỉ lấy của nhân viên đó. Nếu rỗng, lấy tất cả.
        
    Returns:
        list: Các tuple thông tin chấm công.
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    
    query = """
        SELECT c.ma_nv, n.ten_nv, c.ngay, c.gio_vao, c.gio_ra, c.tong_gio
        FROM cham_cong c
        JOIN nhan_vien n ON c.ma_nv = n.ma_nv
        WHERE c.ngay BETWEEN ? AND ?
    """
    params = [tu_ngay, den_ngay]
    
    if ma_nv:
        query += " AND c.ma_nv = ?"
        params.append(ma_nv)
        
    query += " ORDER BY c.ngay DESC, c.id DESC"
        
    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    return rows

def xuat_csv(danh_sach: list, ten_file: str) -> dict:
    """
    Ghi danh sách dữ liệu ra file CSV và lưu thông tin báo cáo.
    
    Args:
        danh_sach (list): Danh sách các hàng dữ liệu cần ghi.
        ten_file (str): Đường dẫn/tên file csv sẽ ghi.
        
    Returns:
        dict: {'thanh_cong': bool, 'thong_bao': str}
    """
    try:
        with open(ten_file, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            # Ghi dòng header
            writer.writerow(["Mã NV", "Họ Tên", "Ngày", "Giờ Vào", "Giờ Ra", "Tổng Giờ"])
            # Ghi từng hàng
            for row in danh_sach:
                # Format dữ liệu để tránh lỗi None
                r = list(row)
                for i in range(len(r)):
                    if r[i] is None:
                        r[i] = ""
                writer.writerow(r)
        return {"thanh_cong": True, "thong_bao": f"Xuất báo cáo thành công ra file {ten_file}!"}
    except Exception as e:
        return {"thanh_cong": False, "thong_bao": f"Lỗi xuất CSV: {str(e)}"}

def lay_ma_nv_tu_id(nv_id: int) -> str:
    """
    Tìm mã nhân viên từ ID số nguyên (dùng cho nhận diện khuôn mặt).
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    cursor.execute("SELECT ma_nv FROM nhan_vien WHERE id = ?", (nv_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def xoa_nhan_vien(ma_nv: str) -> dict:
    """
    Xoá nhân viên khỏi CSDL (và cả dữ liệu chấm công liên quan).
    """
    conn = ket_noi_db()
    cursor = conn.cursor()
    try:
        # Xoá chấm công trước (do ràng buộc FK)
        cursor.execute("DELETE FROM cham_cong WHERE ma_nv = ?", (ma_nv,))
        # Xoá nhân viên
        cursor.execute("DELETE FROM nhan_vien WHERE ma_nv = ?", (ma_nv,))
        conn.commit()
        return {"thanh_cong": True, "thong_bao": f"Đã xoá nhân viên {ma_nv} thành công."}
    except Exception as e:
        return {"thanh_cong": False, "thong_bao": f"Lỗi khi xoá: {str(e)}"}
    finally:
        conn.close()

def in_danh_sach_nhan_vien():
    """
    In ra toàn bộ danh sách nhân viên.
    """
    danh_sach = lay_tat_ca_nhan_vien()
    
    if not danh_sach:
        print("Không có nhân viên nào.")
        return

    print("\n===== DANH SÁCH NHÂN VIÊN =====")
    print(f"{'ID':<5} {'Mã NV':<10} {'Tên NV':<25} {'Phòng Ban'}")
    
    for nv in danh_sach:
        id, ma_nv, ten_nv, phong_ban = nv
        print(f"{id:<5} {ma_nv:<10} {ten_nv:<25} {phong_ban}")

def in_bang_cham_cong():
    """
    In toàn bộ bảng chấm công (kèm ID record).
    """
    conn = ket_noi_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, ma_nv, ngay, gio_vao, gio_ra, tong_gio
        FROM cham_cong
        ORDER BY id DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Chưa có dữ liệu chấm công.")
        return

    print("\n===== BẢNG CHẤM CÔNG =====")
    print(f"{'ID':<5} {'Mã NV':<10} {'Ngày':<12} {'Giờ vào':<10} {'Giờ ra':<10} {'Tổng giờ'}")

    for row in rows:
        id, ma_nv, ngay, gio_vao, gio_ra, tong_gio = row
        
        # tránh None
        gio_vao = gio_vao or ""
        gio_ra = gio_ra or ""
        tong_gio = tong_gio if tong_gio is not None else ""

        print(f"{id:<5} {ma_nv:<10} {ngay:<12} {gio_vao:<10} {gio_ra:<10} {tong_gio}")