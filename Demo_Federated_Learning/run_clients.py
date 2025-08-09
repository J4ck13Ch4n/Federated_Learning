import subprocess
import time
import sys

# Tạo 5 process chạy client.py với ID từ 0 đến 4
processes = []
for i in range(5):
    p = subprocess.Popen([sys.executable, "client.py", str(i)])
    processes.append(p)
    time.sleep(1)

# Đợi tất cả process chạy xong (nếu cần)
for p in processes:
    p.wait()
