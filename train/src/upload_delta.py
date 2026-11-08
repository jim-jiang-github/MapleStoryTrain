"""增量上传：把本地新增的样本（train/reports/delta_manifest.txt）传到服务器对应 train/val 子目录。

多文件 × 8 连接并行，直接 put 单文件（每张 ~1MB，不需要分块）。
用法: python train/upload_delta.py
"""
import threading
import time
from pathlib import Path

import paramiko

HOST, PORT, USER, PWD = "connect.bjb1.seetacloud.com", 33982, "root", "yCJWsjuav2U7"
ROOT = Path(__file__).resolve().parents[2]
YDIR = ROOT / "train" / "data" / "yolo_r2"
REMOTE = "/root/wz_mob_trainkit"
N = 8

man = [l.split("\t") for l in
       (ROOT / "train" / "reports" / "delta_manifest.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
jobs = []
for sub, stem in man:
    img = YDIR / "images" / sub / f"{stem}.png"
    lbl = YDIR / "labels" / sub / f"{stem}.txt"
    if img.exists() and lbl.exists():
        jobs.append((img, f"{REMOTE}/images/{sub}/{img.name}"))
        jobs.append((lbl, f"{REMOTE}/labels/{sub}/{lbl.name}"))
print(f"待传 {len(jobs)} 个文件（{len(man)} 张图 + 标注）", flush=True)

lock = threading.Lock()
done = [0, 0]          # 成功, 失败
t0 = time.time()


def worker():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    sftp = c.open_sftp()
    local = True
    while local:
        with lock:
            local = jobs.pop() if jobs else None
        if not local:
            break
        src, dst = local
        for attempt in range(3):
            try:
                sftp.put(str(src), dst)
                with lock:
                    done[0] += 1
                    if done[0] % 200 == 0:
                        el = time.time() - t0
                        print(f"[{done[0]}/{len(man)*2}] {done[0]/el:.1f} 文件/秒", flush=True)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"FAIL {src.name}: {e}", flush=True)
                    with lock:
                        done[1] += 1
                time.sleep(2)
    sftp.close(); c.close()


ths = [threading.Thread(target=worker) for _ in range(N)]
for t in ths:
    t.start()
for t in ths:
    t.join()
el = time.time() - t0
print(f"完成：成功 {done[0]} · 失败 {done[1]} · 耗时 {el/60:.1f} 分钟", flush=True)
