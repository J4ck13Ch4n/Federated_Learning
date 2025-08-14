import pandas as pd

# Danh sách 3 file cần xử lý
input_files = ["DictionaryBruteForce.pcap_Flow.csv"]

# File xuất ra
output_file = "BruteForce.csv"

# Danh sách chứa DataFrame sau khi sửa
dfs = []

for f in input_files:
    # Đọc file
    df = pd.read_csv(f)

    # Nếu biết tên cột label:
    if "label" in df.columns:
        df["label"] = df["label"].replace("NeedManualLabel", "BruteForce")
    else:
        # Nếu không biết header → label nằm cột cuối
        label_col = df.columns[-1]
        df[label_col] = df[label_col].replace("NeedManualLabel", "BruteForce")

    dfs.append(df)

# Ghép tất cả
merged_df = pd.concat(dfs, ignore_index=True)

# Ghi ra file mới
merged_df.to_csv(output_file, index=False)
print(f"Đã tạo file: {output_file}")
