"""
utils.py
Mục đích: Chứa các hàm tiện ích dùng chung cho toàn bộ ứng dụng như xử lý thời gian, ngày tháng.

"""

import datetime

def lay_gio_hien_tai() -> str:
    """
    Trả về chuỗi giờ hiện tại theo định dạng "HH:MM:SS".
    
    Returns:
        str: Giờ hiện tại (VD: "08:30:15").
    """
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

def lay_ngay_hien_tai() -> str:
    """
    Trả về chuỗi ngày hiện tại theo định dạng "YYYY-MM-DD".
    
    Returns:
        str: Ngày hiện tại (VD: "2026-03-06").
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d")

def tinh_so_gio(gio_vao: str, gio_ra: str) -> float:
    """
    Tính số giờ làm việc giữa 2 mốc thời gian.
    
    Args:
        gio_vao (str): Giờ bắt đầu ở định dạng "HH:MM:SS".
        gio_ra (str): Giờ kết thúc ở định dạng "HH:MM:SS".
        
    Returns:
        float: Tổng số giờ (VD: 8.5 cho 8 tiếng rưỡi).
               Trả về 0.0 nếu có lỗi định dạng thời gian.
    """
    try:
        format_str = "%H:%M:%S"
        t1 = datetime.datetime.strptime(gio_vao, format_str)
        t2 = datetime.datetime.strptime(gio_ra, format_str)
        
        # Nếu giờ ra nhỏ hơn giờ vào (ví dụ làm qua đêm), có thể cộng thêm 1 ngày
        if t2 < t1:
            t2 += datetime.timedelta(days=1)
            
        diff = t2 - t1
        # Trả về số giờ tính bằng decimal, làm tròn 2 chữ số
        return round(diff.total_seconds() / 3600.0, 2)
    except ValueError:
        return 0.0

def dinh_dang_hien_thi(so_gio: float) -> str:
    """
    Chuyển đổi số giờ thành chuỗi thân thiện dễ đọc.
    
    Args:
        so_gio (float): Tổng số giờ dạng thập phân.
        
    Returns:
        str: Chuỗi định dạng (VD: 8.5 -> "8 giờ 30 phút").
    """
    if so_gio is None:
        return ""
    
    gio = int(so_gio)
    phut = int(round((so_gio - gio) * 60))
    
    if phut == 60:
        gio += 1
        phut = 0
        
    return f"{gio} giờ {phut} phút"
