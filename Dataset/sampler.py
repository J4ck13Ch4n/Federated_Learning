import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# --- Cấu hình Sampling ---
# Hướng dẫn:
# 1. Chọn phương pháp bạn muốn trong SAMPLING_METHOD.
#    Các lựa chọn: 'random', 'stratified', 'oversample', 'undersample'
# 2. Thiết lập các tham số tương ứng bên dưới.

SAMPLING_METHOD = 'stratified'  # <-- THAY ĐỔI PHƯƠNG PHÁP Ở ĐÂY

# --- Tham số cho 'random' và 'stratified' ---
# - Điền một số từ 0.0 đến 1.0 để lấy theo tỷ lệ (ví dụ: 0.5 là 50% dữ liệu).
# - Điền một số nguyên để lấy số lượng mẫu chính xác (ví dụ: 10000 là 10,000 mẫu).
SAMPLE_SIZE = 0.5 # <-- THAY ĐỔI TỶ LỆ/SỐ LƯỢNG Ở ĐÂY

# --- Bước 1: Tải dữ liệu ---
print("Đang tải dữ liệu từ IoTDIAD_sum.csv...")
try:
    df = pd.read_csv("f:/NCKH_new/Federated_Learning/Dataset/IoTDIAD_sum.csv")
    print("Tải dữ liệu thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'IoTDIAD_sum.csv'.")
    print("Vui lòng chạy file 'data_sum.py' trước để tạo file dữ liệu tổng hợp.")
    exit()

# --- Bước 2: Phân tích dữ liệu ban đầu ---
print(f"\nKích thước dữ liệu gốc: {df.shape}")
target_column = df.columns[-1]
print(f"Cột mục tiêu được xác định là: '{target_column}'")
print("\nPhân phối lớp ban đầu:")
print(df[target_column].value_counts())

# --- Bước 3: Tiền xử lý (cần cho over/under-sampling) ---
# Sao chép để không ảnh hưởng đến dataframe gốc khi sampling đơn giản
processed_df = df.copy()
categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"\nPhát hiện các cột không phải số: {categorical_cols}. Đang tiến hành Label Encoding...")
    for col in categorical_cols:
        processed_df[col] = LabelEncoder().fit_transform(processed_df[col].astype(str))

processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
processed_df.fillna(0, inplace=True)
print("Đã hoàn tất tiền xử lý (encoding và xử lý NaN/inf).")

X = processed_df.drop(columns=[target_column])
y = processed_df[target_column]

# --- Bước 4: Áp dụng phương pháp Sampling đã chọn ---
print(f"\nĐang áp dụng phương pháp: '{SAMPLING_METHOD}'...")

sampled_df = None

if SAMPLING_METHOD == 'random':
    if 0.0 < SAMPLE_SIZE <= 1.0:
        sampled_df = df.sample(frac=SAMPLE_SIZE, random_state=42)
    elif SAMPLE_SIZE > 1:
        sampled_df = df.sample(n=int(SAMPLE_SIZE), random_state=42)
    else:
        raise ValueError("SAMPLE_SIZE cho 'random' phải là tỉ lệ (0-1) hoặc số nguyên > 1.")

elif SAMPLING_METHOD == 'stratified':
    if 0.0 < SAMPLE_SIZE < 1.0:
        # Sử dụng train_test_split để thực hiện stratified sampling
        _, X_sampled, _, y_sampled = train_test_split(X, y, test_size=SAMPLE_SIZE, stratify=y, random_state=42)
        sampled_df = pd.concat([X_sampled, y_sampled], axis=1)
        # Khôi phục lại các cột object nếu có
        for col in categorical_cols:
            sampled_df[col] = df[col]
    else:
        raise ValueError("SAMPLE_SIZE cho 'stratified' phải là một tỉ lệ từ 0.0 đến 1.0.")


elif SAMPLING_METHOD == 'oversample':
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    sampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    sampled_df[target_column] = y_resampled

elif SAMPLING_METHOD == 'undersample':
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    sampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    sampled_df[target_column] = y_resampled

else:
    print(f"Lỗi: Phương pháp '{SAMPLING_METHOD}' không được hỗ trợ.")
    exit()

print("Áp dụng sampling thành công!")

# --- Bước 5: Lưu và kiểm tra kết quả ---
if sampled_df is not None:
    output_path = "f:/NCKH_new/Federated_Learning/Dataset/IoTDIAD_sampled.csv"
    sampled_df.to_csv(output_path, index=False)

    print(f"\nKích thước dữ liệu sau khi sampling: {sampled_df.shape}")
    print("\nPhân phối lớp sau khi sampling:")
    print(sampled_df[target_column].value_counts())
    print(f"\nĐã lưu dữ liệu đã được sampling vào file '{output_path}'")
else:
    print("\nKhông có dữ liệu nào được tạo ra.")

print("\nHoàn thành!")
