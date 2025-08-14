import pandas as pd
import os

base_dir = "f:/NCKH_new/Federated_Learning/Dataset"
folders = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"]
csv_files = [os.path.join(base_dir, folder, f"{folder if folder != 'Web-based' else 'WebBased'}.csv") for folder in folders]

dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    if len(df) >= 25000:
        df_sample = df.sample(n=25000, random_state=42)
    else:
        df_sample = df
    dfs.append(df_sample)

# Gộp lại
full_df = pd.concat(dfs, ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Lưu ra file tổng hợp
full_df.to_csv(f"{base_dir}/IoTDIAD_sum.csv", index=False)

print("Đã tạo file tổng hợp với tối đa 25,000 record mỗi class.")