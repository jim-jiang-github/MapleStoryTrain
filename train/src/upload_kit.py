# -*- coding: utf-8 -*-
"""上传训练包到远程服务器（SFTP，带断点续传 + 实时测速）"""
import sys, time, os
import paramiko

HOST, PORT, USER, PWD = "region-41.seetacloud.com", 34660, "root", "NHjO9fbz6org"
LOCAL = os.path.join(os.path.dirname(__file__), "..", "kit", "wz_mob_trainkit.zip")
REMOTE = "/root/wz_mob_trainkit_v2.zip"  # 服务器已有旧文件，换名不覆盖

LOCAL = os.path.abspath(LOCAL)
total = os.path.getsize(LOCAL)
print(f"本地文件: {LOCAL} ({total/1e9:.2f} GB)", flush=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
sftp = c.open_sftp()

# 断点续传：检查远端已有大小
try:
    rsize = sftp.stat(REMOTE).st_size
except FileNotFoundError:
    rsize = 0
if rsize > total:
    print("远端文件比本地大，删除重传", flush=True)
    sftp.remove(REMOTE); rsize = 0
if rsize == total:
    print("远端已是完整文件，无需上传", flush=True); sys.exit(0)
print(f"从 offset {rsize/1e9:.2f} GB 续传" if rsize else "全新上传", flush=True)

t0 = time.time()
last_t, last_b = t0, rsize
window_start, window_bytes = t0, 0   # 滚动测速窗口

def cb(sent, tot):
    global last_t, last_b, window_start, window_bytes
    now = time.time()
    window_bytes = sent - rsize
    if now - last_t >= 15:  # 每 15 秒报一次
        dt = now - last_t
        inst = (sent - last_b) / dt / 1e6
        elapsed = now - t0
        avg = window_bytes / elapsed / 1e6 if elapsed > 0 else 0
        eta = (tot - sent) / (avg * 1e6) if avg > 0 else 0
        print(f"[{sent/tot*100:5.1f}%] {sent/1e9:5.2f}/{tot/1e9:.2f} GB | "
              f"瞬时 {inst:6.1f} MB/s | 平均 {avg:6.1f} MB/s | "
              f"剩余 {eta/60:4.1f} 分钟", flush=True)
        last_t, last_b = now, sent

with open(LOCAL, "rb") as f:
    f.seek(rsize)
    if rsize:
        rf = sftp.open(REMOTE, "ab")
    else:
        rf = sftp.open(REMOTE, "wb")
    rf.set_pipelined(True)
    try:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            rf.write(chunk)
            cb(rsize + f.tell(), total)
        rf.flush()
    finally:
        rf.close()

# 校验
rsize = sftp.stat(REMOTE).st_size
ok = rsize == total
dt = time.time() - t0
print(f"\n上传完成: 远端 {rsize/1e9:.2f} GB / 本地 {total/1e9:.2f} GB "
      f"{'✓ 一致' if ok else '✗ 大小不符!'}", flush=True)
if ok:
    print(f"本次耗时 {dt/60:.1f} 分钟, 整体平均速度 {(total-rsize+total if False else window_bytes)/dt/1e6:.1f} MB/s", flush=True)
sftp.close(); c.close()
sys.exit(0 if ok else 1)
