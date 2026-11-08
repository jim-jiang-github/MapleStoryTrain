# -*- coding: utf-8 -*-
"""并行分块上传训练包：N 条 SSH 连接各传一段，最后远端拼接。带断点续传 + 汇总测速。"""
import sys, os, time, threading
import paramiko

HOST, PORT, USER, PWD = "region-41.seetacloud.com", 34660, "root", "NHjO9fbz6org"
LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kit", "wz_mob_trainkit.zip"))
REMOTE = "/root/wz_mob_trainkit_v2.zip"
PARTS_DIR = "/root/_wk_parts"
N = 8  # 并行流数

total = os.path.getsize(LOCAL)
chunk = (total + N - 1) // N
print(f"本地 {total/1e9:.2f} GB → {N} 路并行, 每路 {chunk/1e9:.2f} GB → 远端 {PARTS_DIR}/ → 拼成 {REMOTE}", flush=True)

# 连接工厂
def connect():
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PWD)
    return t

# 远端准备：清掉旧的部分文件和残留单流文件
tr = connect(); s0 = paramiko.SFTPClient.from_transport(tr)
for f in s0.listdir("/root"):
    if f == "_wk_parts" or f in ("wz_mob_trainkit.zip", "wz_mob_trainkit_v2.zip"):
        pass
s0.mkdir(PARTS_DIR) if PARTS_DIR.split("/")[-1] not in s0.listdir("/root") else None
for p in s0.listdir(PARTS_DIR):
    s0.remove(PARTS_DIR + "/" + p)
try: s0.remove(REMOTE)          # 上一个单流版的半截文件
except FileNotFoundError: pass
s0.close(); tr.close()
print("远端已清理旧部分文件", flush=True)

# 统计
lock = threading.Lock()
progress = [0] * N            # 每路已传字节
bytes_at_start = [0] * N
t0 = time.time()

def upload_part(i):
    lo = i * chunk
    hi = min((i + 1) * chunk, total)
    part = f"{PARTS_DIR}/part{i:02d}"
    tr = connect()
    sftp = paramiko.SFTPClient.from_transport(tr)
    try:
        rsize = sftp.stat(part).st_size
    except FileNotFoundError:
        rsize = 0
    # 越界保护
    if rsize > hi - lo:
        sftp.remove(part); rsize = 0
    bytes_at_start[i] = rsize
    progress[i] = rsize
    if rsize == hi - lo:
        print(f"[part{i:02d}] 已完整, 跳过", flush=True)
        sftp.close(); tr.close(); return
    rf = sftp.open(part, "ab" if rsize else "wb")
    rf.set_pipelined(True)
    with open(LOCAL, "rb") as f:
        f.seek(lo + rsize)
        pos = rsize
        try:
            while True:
                need = min(4 * 1024 * 1024, (hi - lo) - pos)
                if need <= 0: break
                data = f.read(need)
                if not data: break
                rf.write(data)
                pos += len(data)
                with lock:
                    progress[i] = pos
        except Exception as e:
            print(f"[part{i:02d}] 出错: {e}", flush=True)
            raise
    rf.flush(); rf.close(); sftp.close(); tr.close()

def reporter():
    def sz(i):  # 第 i 块的大小
        return min((i + 1) * chunk, total) - i * chunk
    while any(progress[i] < sz(i) for i in range(N)):
        time.sleep(15)
        with lock:
            done = sum(bytes_at_start) + sum(progress)
        el = time.time() - t0
        sent = sum(progress) - sum(bytes_at_start)
        avg = sent / el / 1e6 if el > 0 else 0
        eta = (total - done) / (avg * 1e6) if avg > 0 else 0
        per = " ".join(f"p{i}:{progress[i]/sz(i)*100:3.0f}%" for i in range(N))
        print(f"[{done/total*100:5.1f}%] {done/1e9:5.2f}/{total/1e9:.2f} GB | "
              f"合计 {avg:6.1f} MB/s | 剩余 {eta/60:4.1f} 分钟 | {per}", flush=True)

rep = threading.Thread(target=reporter, daemon=True)
rep.start()

ths = [threading.Thread(target=upload_part, args=(i,)) for i in range(N)]
t0 = time.time()
for t in ths: t.start()
errs = []
for t in ths: t.join()

def sz(i):
    return min((i + 1) * chunk, total) - i * chunk
if any(t.is_alive() for t in ths) or any(progress[i] < sz(i) for i in range(N)):
    print("\n有分块未完成，重跑本脚本可断点续传", flush=True)
    sys.exit(1)

el = time.time() - t0
print(f"\n全部分块上传完成, 耗时 {el/60:.1f} 分钟, 平均 {(total-sum(bytes_at_start))/el/1e6:.1f} MB/s", flush=True)

# 远端拼接 + 校验
print("远端拼接中...", flush=True)
tr = connect()
_, out, err = tr.open_session()
out.exec_command(f"cd /root && rm -f {REMOTE} && cat {PARTS_DIR}/part* > {REMOTE} && ls -l {REMOTE} && md5sum {REMOTE}")
print(out.read().decode().strip(), err.read().decode().strip(), sep="\n", flush=True)

import hashlib
h = hashlib.md5()
with open(LOCAL, "rb") as f:
    for blk in iter(lambda: f.read(64 * 1024 * 1024), b""):
        h.update(blk)
print("本地 md5:", h.hexdigest(), flush=True)
tr.close()
print("\n对比两行 md5 一致即上传完整。之后可 rm -rf /root/_wk_parts 释放 13G（脚本下次可代做）", flush=True)
