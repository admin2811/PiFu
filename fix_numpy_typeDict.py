import numpy as np
import sys
import scipy
from pathlib import Path

# Tìm đường dẫn đến file sputils.py
scipy_path = Path(scipy.__file__).parent
sputils_path = scipy_path / 'sparse' / 'sputils.py'

# Đọc nội dung file gốc
with open(sputils_path, 'r') as f:
    content = f.read()

# Thay thế dòng có numpy.typeDict
new_content = content.replace(
    "supported_dtypes = [np.typeDict[x] for x in supported_dtypes]",
    "supported_dtypes = [np.dtype(x) for x in supported_dtypes]"
)

# Ghi đè lên file gốc
with open(sputils_path, 'w') as f:
    f.write(new_content)

print("Đã sửa xong lỗi numpy.typeDict") 