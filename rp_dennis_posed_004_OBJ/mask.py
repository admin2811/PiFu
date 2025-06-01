import cv2
import numpy as np
from PIL import Image

# Đọc ảnh texture
image_path = 'D:/DoAnTotNghiep/DoAn/PIFu/cartoon/tex/rp_dennis_posed_004_dif_8k.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ===================== 1. MASK MÀU DA (VÀNG) =====================
lower_yellow = np.array([200, 150, 80])
upper_yellow = np.array([255, 220, 150])
mask_yellow = cv2.inRange(img_rgb, lower_yellow, upper_yellow)
mask_yellow[mask_yellow > 0] = 255
Image.fromarray(mask_yellow).save('D:/DoAnTotNghiep/DoAn/PIFu/cartoon/tex/rp_dennis_posed_004_spec_skin01.tif.tif')

# ===================== 2. MASK MÀU ĐỎ =====================
lower_red = np.array([180, 60, 60])
upper_red = np.array([255, 150, 130])
mask_red = cv2.inRange(img_rgb, lower_red, upper_red)
mask_red[mask_red > 0] = 255
Image.fromarray(mask_red).save('D:/DoAnTotNghiep/DoAn/PIFu/cartoon/tex/rp_dennis_posed_004_spec_shirt_fabric02.tif')

# ===================== 3. MASK MÀU TRẮNG (VÙNG SÁNG) =====================
lower_white = np.array([240, 240, 220])
upper_white = np.array([255, 255, 255])
mask_white = cv2.inRange(img_rgb, lower_white, upper_white)
mask_white[mask_white > 0] = 255
Image.fromarray(mask_white).save('D:/DoAnTotNghiep/DoAn/PIFu/cartoon/tex/rp_dennis_posed_004_spec_hair01.tif')

# ===================== 4. MASK PHẦN CÒN LẠI (KHÔNG PHẢI 3 LOẠI TRÊN) =====================
# Gộp tất cả mask lại
combined_mask = cv2.bitwise_or(mask_yellow, mask_red)
combined_mask = cv2.bitwise_or(combined_mask, mask_white)

# Phần còn lại = không thuộc mask nào ở trên
mask_other = cv2.bitwise_not(combined_mask)
mask_other[mask_other > 0] = 255
Image.fromarray(mask_other).save('D:/DoAnTotNghiep/DoAn/PIFu/cartoon/tex/rp_dennis_posed_004_spec_shoes_leather01.tif')

print("✅ Đã tạo 4 file mask riêng biệt thành công!")
