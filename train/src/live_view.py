# -*- coding: utf-8 -*-
"""实时跟随远程训练日志：总进度条 + 本轮进度条 + 倒计时"""
import sys, time, re, datetime
import paramiko

HOST, PORT, USER, PWD = "connect.bjb1.seetacloud.com", 33982, "root", "yCJWsjuav2U7"
LOG_CAND = ["/root/wz_mob_trainkit/finetune.log", "/root/wz_mob_trainkit/train.log"]
EPOCHS = 100  # 会被日志里的 "n/总" 动态覆盖

# ---------------- 状态 ----------------
S = {
    "epoch": 0, "total_ep": 100, "batch": 0, "total": 0, "its": 0.0,
    "box": 0.0, "cls": 0.0, "dfl": 0.0, "mem": 0.0, "inst": 0,
    "t_el": 0.0, "t_rem": None,        # 本轮已用 / 本轮剩余（秒）
    "remain_at": None, "remain_sec": None,  # 总剩余（含倒计时锚点）
    "t0": time.time(), "epoch_durs": [], "epoch_marks": {},
    "metrics": "尚无（第一轮验证还在跑）",
    "history": [], "init": True,       # init 阶段不记录单轮耗时（历史一次性灌入）
}


def sec(v):
    """'1:11' / '1:11:22' / '59.5s' / '2:02' -> 秒"""
    if v is None:
        return None
    v = str(v).strip()
    if v.endswith("s"):
        try:
            return float(v[:-1])
        except ValueError:
            return None
    parts = v.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    t = 0.0
    for n in nums:
        t = t * 60 + n
    return t


def hms(t, short=False):
    t = max(0, int(round(t)))
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    if short and h == 0:
        return f"{m}m{s:02d}s"
    return f"{h}:{m:02d}:{s:02d}"


def bar(frac, width=28):
    frac = max(0.0, min(1.0, frac))
    n = int(frac * width)
    return "█" * n + "░" * (width - n)


def feed(seg, ep_label=None):
    """解析一行日志，更新状态"""
    if re.search(r"\d+/\d+", seg) and "it/s" in seg:
        m = re.search(r"(\d+)/(\d+)\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+\d{3,4}:?", seg)
        p = re.search(r"(\d+)/(\d+)\s+([\d.]+)it/s\s+([\d:.]+)(?:<([\d:.]+))?", seg)
        if not p:
            return
        if m:
            S["total_ep"] = int(m.group(2))
        ep = int(m.group(1)) if m else S["epoch"]
        if ep != S["epoch"]:
            now = time.time()
            if not S["init"]:              # 只统计实时观测到的轮次切换
                if ep in S["epoch_marks"]:
                    d = now - S["epoch_marks"][ep]
                    if 60 < d < 3600:
                        S["epoch_durs"].append(d)
                        S["epoch_durs"] = S["epoch_durs"][-5:]
                S["epoch_marks"][ep] = now
        S["epoch"] = ep
        if m:
            S["mem"], S["box"], S["cls"], S["dfl"], S["inst"] = (
                float(m.group(3)), float(m.group(4)), float(m.group(5)),
                float(m.group(6)), int(m.group(7)))
        S["batch"], S["total"], S["its"] = int(p.group(1)), int(p.group(2)), float(p.group(3))
        S["t_el"] = sec(p.group(4)) or 0.0
        S["t_rem"] = sec(p.group(5))

        # 单轮耗时：主用「批次数 ÷ 速度 + 验证开销」，有实时实测值就用实测值
        it = S["its"] if S["its"] > 0.5 else 9.0
        dur_calc = S["total"] / it + 18
        if S["t_rem"] is not None and S["t_el"] > 0:
            dur_calc = max(dur_calc, S["t_el"] + S["t_rem"] + 18)
        if S["epoch_durs"]:
            dur = sum(S["epoch_durs"]) / len(S["epoch_durs"])
        else:
            dur = dur_calc
        done_ep = (S["epoch"] - 1) + S["batch"] / max(1, S["total"])
        left_ep = max(0.0, S.get("total_ep", EPOCHS) - done_ep)
        S["remain_sec"] = left_ep * dur
        S["remain_at"] = time.time()
    elif re.search(r"\ball\s+\d+", seg):
        nums = re.findall(r"[-\d.]+", seg)
        if len(nums) >= 4:
            p4 = [float(x) for x in nums[-4:]]      # 最后 4 个数：P R mAP50 mAP50-95
            S["metrics"] = (f"P {p4[0]:.3f}   R {p4[1]:.3f}   "
                            f"mAP50 {p4[2]:.3f}   mAP50-95 {p4[3]:.3f}")
            ep = ep_label if ep_label is not None else S["epoch"]
            S["history"].append(f"epoch {str(ep):>3}  " + S["metrics"])
            S["history"] = S["history"][-3:]
    elif any(k in seg for k in ("Results saved", "epochs completed", "optimizer stripped")):
        S["history"].append(seg.strip()[:96])


def render():
    now = time.time()
    out = []
    done_ep = (S["epoch"] - 1) + (S["batch"] / S["total"] if S["total"] else 0)
    tot = S.get("total_ep", EPOCHS)
    ov = done_ep / tot
    if S["remain_sec"] is not None:
        left = max(0.0, S["remain_sec"] - (now - S["remain_at"]))
        eta = datetime.datetime.fromtimestamp(now + left).strftime("%H:%M:%S")
        left_s, eta_s = hms(left), eta
    else:
        left_s, eta_s = "计算中…", "--:--:--"
    ep_frac = (S["batch"] / S["total"]) if S["total"] else 0

    out.append("=" * 78)
    out.append(f"  WZ 怪物检测训练 · 实时监控 (RTX 4090)        已运行 {hms(now - S['t0'])}")
    out.append("=" * 78)
    out.append(f"  总进度 [{bar(ov)}] {ov*100:5.1f}%   epoch {S['epoch']}/{tot}")
    out.append(f"  本轮   [{bar(ep_frac)}] {ep_frac*100:5.1f}%   "
               f"{S['batch']}/{S['total']} 批   {S['its']:.1f} it/s")
    out.append("-" * 78)
    out.append(f"  loss  box {S['box']:.3f}   cls {S['cls']:.3f}   dfl {S['dfl']:.3f}"
               f"     显存 {S['mem']:.1f}G   目标 {S['inst']}")
    out.append("-" * 78)
    out.append(f"  ⏳ 倒计时  {left_s}      预计完成 {eta_s}")
    out.append("-" * 78)
    out.append(f"  最新验证  {S['metrics']}")
    for h in S["history"][-3:]:
        out.append(f"    {h}")
    out.append("-" * 78)
    out.append("  Ctrl+C 退出监控（不影响服务器训练）")
    # 回到顶部，逐行覆盖并清尾
    sys.stdout.write("\033[H")
    for line in out:
        sys.stdout.write(line[:200] + "\033[K\n")
    sys.stdout.flush()


# ---------------- 主流程 ----------------
sys.stdout.write("\033[2J\033[H")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)

# 先补一次历史（拿初始状态，避免刚连上时空白），同时取训练真实开始时间
# 选最新修改的日志（微调 finetune.log 优先于首训 train.log）
_, o, _ = c.exec_command("ls -t " + " ".join(LOG_CAND) + " | head -1", timeout=30)
LOG = o.read().decode().strip() or LOG_CAND[-1]
_, o, _ = c.exec_command(
    f"tail -c 20000 {LOG} | tr '\\r' '\\n'; echo '@@@MARK@@@'; "
    f"grep -a ' all ' {LOG} | tail -3; echo '@@@MARK@@@'; "
    f"find /root/wz_mob_trainkit/runs/detect -maxdepth 3 -name args.yaml -printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -1",
    timeout=30)
raw = o.read().decode("utf-8", "replace").split("@@@MARK@@@")
for line in raw[0].splitlines():
    if line.strip():
        feed(line)
# 历史验证指标（从旧到新，按 epoch 倒推编号）
allines = [l for l in raw[1].splitlines() if l.strip()]
cur = S["epoch"]
for i, l in enumerate(allines):
    feed(l, ep_label=max(1, cur - (len(allines) - i)))
S["init"] = False
try:
    start_ts = float(raw[2].strip().splitlines()[0].split()[0])
    S["t0"] = start_ts
except Exception:
    S["t0"] = time.time() - (S["t_el"] or 0)

ch = c.get_transport().open_session()
ch.exec_command(f"tail -n 0 -f {LOG}")
ch.settimeout(0.3)

buf = ""
last = 0.0
try:
    render()
    while True:
        try:
            data = ch.recv(65536)
            if not data:
                break
            buf += data.decode("utf-8", "replace")
        except Exception:
            pass
        while True:
            m = re.search(r"[\r\n]", buf)
            if not m:
                break
            seg, buf = buf[:m.start()].rstrip(), buf[m.end():]
            if seg:
                feed(seg)
        if time.time() - last > 0.4:      # 倒计时每秒都在走
            render()
            last = time.time()
except KeyboardInterrupt:
    pass
finally:
    sys.stdout.write("\n\n退出监控（训练仍在服务器上继续）\n")
    ch.close(); c.close()
