# -*- coding: utf-8 -*-
"""每5分钟轮询服务器训练进度，记录 epoch 转换时间戳，计算每个epoch实际耗时"""
import paramiko, time, re, sys

HOST, PORT, USER, PWD = "region-41.seetacloud.com", 34660, "root", "NHjO9fbz6org"
LOG = "/root/wz_mob_trainkit/train.log"
OUT = "train/epoch_speed.log"

def poll():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    _, out, _ = c.exec_command(
        f"tail -c 3000 {LOG} | tr '\\r' '\\n' | grep -a '/100' | tail -1; "
        f"ps aux | grep -v grep | grep -c 'train.py' || true", timeout=30)
    txt = out.read().decode(errors="replace").strip()
    c.close()
    alive = txt.splitlines()[-1].strip() if txt else "0"
    m = re.search(r"(\d+)/100.*?([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+).*?(\d+)/(\d+)\s+([\d.]+)it/s", txt)
    return m, alive

# epoch -> 首次看到的时间
seen = {}
while True:
    try:
        m, alive = poll()
        now = time.strftime("%H:%M:%S")
        if m:
            ep = int(m.group(1)); mem = m.group(2)
            box, cls, dfl = m.group(3), m.group(4), m.group(5)
            batch, total = m.group(6), m.group(7)
            rate = float(m.group(8))
            if ep not in seen:
                seen[ep] = time.time()
                # 计算上一轮耗时
                line = f"[{now}] epoch {ep}/100 开始"
                if ep - 1 in seen:
                    dt = (seen[ep] - seen[ep-1]) / 60
                    line += f"（epoch {ep-1} 耗时 {dt:.1f} 分钟）"
                print(line, flush=True)
            print(f"[{now}] epoch {ep}/100  batch {batch}/{total}  {rate:.1f} it/s  "
                  f"预计本epoch还剩 {(int(total)-int(batch))/rate/60:.1f} 分钟  "
                  f"loss: box={box} cls={cls} dfl={dfl}  GPU {mem}G", flush=True)
        if alive == "0":
            print(f"[{now}] 训练进程已结束", flush=True)
            break
    except Exception as e:
        print(f"轮询出错: {e}", flush=True)
    time.sleep(300)
