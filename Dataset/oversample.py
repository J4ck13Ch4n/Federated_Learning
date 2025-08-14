import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np

# Chào bạn, để thực hiện oversampling trên dữ liệu có các feature không phải là số,
# chúng ta cần thực hiện các bước sau:
# 1. Tải dữ liệu từ file CSV đã được tổng hợp.
# 2. Xác định các cột dữ liệu không phải là số (categorical/object) và cột mục tiêu (target).
# 3. Chuyển đổi các cột không phải số thành dạng số. Label Encoding là một cách đơn giản,
#    nhưng One-Hot Encoding thường cho kết quả tốt hơn. Ở đây chúng ta sẽ dùng LabelEncoder
#    để xử lý các giá trị NaN và vô cực trước.
# 4. Xử lý các giá trị vô cực (infinity) có thể phát sinh trong dữ liệu.
# 5. Tách dữ liệu thành features (X) và target (y).
# 6. Áp dụng kỹ thuật oversampling SMOTE để cân bằng dữ liệu.
# 7. In ra số lượng mẫu của mỗi lớp trước và sau khi oversampling để kiểm tra.

# --- Bước 1: Tải dữ liệu ---
print("Đang tải dữ liệu từ IoTDIAD_sum.csv...")
try:
    df = pd.read_csv("f:/NCKH_new/Federated_Learning/Dataset/IoTDIAD_sum.csv")
    print("Tải dữ liệu thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'IoTDIAD_sum.csv'.")
    print("Vui lòng chạy file 'data_sum.py' trước để tạo file dữ liệu tổng hợp.")
    exit()

# --- Bước 2: Xác định và xử lý các cột không phải số ---
print("\nBắt đầu xử lý dữ liệu...")
# Giả sử cột cuối cùng là cột target 'label'
target_column = df.columns[-1]
print(f"Cột mục tiêu được xác định là: '{target_column}'")

# In ra số lượng mẫu mỗi lớp TRƯỚC khi oversampling
print("\nSố lượng mẫu mỗi lớp (trước khi oversampling):")
print(df[target_column].value_counts())

# Xác định các cột có kiểu dữ liệu 'object' (thường là chuỗi)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# --- Bước 3: Chuyển đổi cột không phải số thành số ---
if categorical_cols:
    print(f"\nCác cột không phải số cần chuyển đổi: {categorical_cols}")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Chuyển đổi cột sang kiểu string và điền giá trị thiếu nếu có
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    print("Đã chuyển đổi các cột không phải số thành dạng số bằng LabelEncoder.")
else:
    print("\nKhông tìm thấy cột nào cần chuyển đổi.")

# --- Bước 4: Xử lý giá trị vô cực và NaN ---
# Thay thế giá trị vô cực bằng một số lớn (hoặc NaN rồi điền giá trị trung bình/median)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Điền các giá trị NaN còn lại bằng 0 hoặc giá trị trung bình/median
df.fillna(0, inplace=True)
print("Đã xử lý các giá trị vô cực (infinity) và NaN.")

# --- Bước 5: Tách features (X) và target (y) ---
X = df.drop(columns=[target_column])
y = df[target_column]
print("Đã tách xong features (X) và target (y).")

# --- Bước 6: Áp dụng SMOTE ---
# Để chạy được bước này, bạn cần cài đặt thư viện 'imbalanced-learn'
# Mở terminal và chạy lệnh: pip install imbalanced-learn
print("\nBắt đầu áp dụng SMOTE để oversampling...")
try:
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Áp dụng SMOTE thành công.")
except Exception as e:
    print(f"Lỗi khi áp dụng SMOTE: {e}")
    print("Hãy chắc chắn rằng bạn đã cài đặt thư viện 'imbalanced-learn'.")
    print("Chạy lệnh sau trong terminal: pip install imbalanced-learn")
    exit()

# --- Bước 7: Kiểm tra kết quả ---
print("\nSố lượng mẫu mỗi lớp (SAU khi oversampling):")
# y_resampled là một Series của pandas, có thể dùng value_counts()
print(y_resampled.value_counts())

# (Tùy chọn) Lưu dữ liệu đã được cân bằng ra file mới
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df[target_column] = y_resampled
resampled_df.to_csv("f:/NCKH_new/Federated_Learning/Dataset/IoTDIAD_resampled.csv", index=False)
print("\nĐã lưu dữ liệu đã được cân bằng vào file 'IoTDIAD_resampled.csv'")

print("\nHoàn thành! Dữ liệu của bạn đã được cân bằng.")
