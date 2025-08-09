import subprocess
import sys
import time
from typing import List


def main():
    # args: num_clients [delay_seconds]
    if len(sys.argv) < 2:
        print("Usage: python client_spawner.py <num_clients> [delay_seconds]")
        sys.exit(1)

    num_clients = int(sys.argv[1])
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5

    time.sleep(delay)

    procs: List[subprocess.Popen] = []
    for i in range(num_clients):
        p = subprocess.Popen([sys.executable, "client.py", str(i)])
        procs.append(p)
        time.sleep(0.5)

    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
