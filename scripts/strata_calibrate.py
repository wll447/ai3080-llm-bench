#!/usr/bin/env python3
"""ai3080: 跑官方调参器 ./setup.sh --calibrate，并前后对照。全流程一次跑完，输出留给人工判读。

流程：备份配置 -> 停掉当前实例（标定要占 8080）-> ./setup.sh --calibrate --no-start
      -> 对比配置差异（只该动 pcie-frac / spec-min-p / pool-workers 那几个）
      -> 启动 -> HTTP 探测就绪 -> 快速复测（2 次 decode + 1 次 ~31k prefill，提示均唯一）
"""
import json
import pathlib
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid

HOME = pathlib.Path.home()
STR = HOME / "Strata"
CFG = STR / "strata-unsloth-ud-q4_k_xl.json"
BAK = STR / "strata-unsloth-ud-q4_k_xl.json.bak-before-calibrate"
LOG = STR / "strata-unsloth-ud-q4_k_xl.log"
SESSION = "strata-262k"
API = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH = "http://127.0.0.1:8080/health"
PARA = ("The harbour lights were dim that evening, and the fog rolled in from the north-east in slow grey banks. "
        "Sailors say the beam must sweep the water twice before a boat may turn, and the keeper counts the turns aloud. "
        "Below the tower, the rocks are black and wet, and the tide makes a sound like breathing in a sleeping room. ")


def ready():
    try:
        with urllib.request.urlopen(HEALTH, timeout=5) as r:
            return '"loaded": true' in r.read().decode()
    except Exception:
        return False


def post(body, timeout=1200):
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, time.time() - t0


def stop():
    for pat in ("[s]erve/server.py", "[e]ngine/strata"):
        subprocess.run(["pkill", "-INT", "-f", pat], check=False)
    for _ in range(20):
        if subprocess.run(["pgrep", "-f", "[s]erve/server.py"], capture_output=True).returncode != 0:
            break
        time.sleep(2)
    subprocess.run(["pkill", "-TERM", "-f", "[s]erve/server.py"], check=False)
    subprocess.run(["pkill", "-TERM", "-f", "[e]ngine/strata"], check=False)
    time.sleep(3)


def start():
    subprocess.run(["tmux", "kill-session", "-t", SESSION], check=False)
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION,
                    "cd %s && exec ./run-unsloth-ud-q4_k_xl.sh" % STR], check=False)


def main():
    shutil.copy2(CFG, BAK)
    before = json.loads(BAK.read_text(encoding="utf-8"))
    print("=== 备份: %s ===" % BAK.name)
    print("=== 标定前 args: %s" % " ".join(before["args"]))

    print("=== 停服（标定要占 8080）===")
    stop()

    print("=== 跑 ./setup.sh --calibrate --no-start（5-10 分钟）===")
    t0 = time.time()
    p = subprocess.run(["./setup.sh", "--calibrate", "--no-start"], cwd=str(STR),
                       capture_output=True, text=True, timeout=2400)
    print("=== setup.sh 退出码 %d，用时 %.0f s ===" % (p.returncode, time.time() - t0))
    for line in (p.stdout or "").splitlines()[-40:]:
        print("  " + line[:200])
    if p.stderr:
        print("--- stderr ---")
        for line in p.stderr.splitlines()[-15:]:
            print("  " + line[:200])

    after = json.loads(CFG.read_text(encoding="utf-8"))
    print("=== 标定后 args: %s" % " ".join(after["args"]))
    ba, aa = before["args"], after["args"]
    diffs = []
    keys = set(ba[::2]) | set(aa[::2])
    for k in sorted(keys):
        vb = ba[ba.index(k) + 1] if k in ba and ba.index(k) + 1 < len(ba) and not ba[ba.index(k) + 1].startswith("--") else "(flag)"
        va = aa[aa.index(k) + 1] if k in aa and aa.index(k) + 1 < len(aa) and not aa[aa.index(k) + 1].startswith("--") else "(flag)"
        if vb != va:
            diffs.append("%s: %s -> %s" % (k, vb, va))
    print("=== 参数差异 ===")
    for d in diffs or ["（无差异）"]:
        print("  " + d)

    print("=== 启动并等待就绪 ===")
    start()
    t0 = time.time()
    while time.time() - t0 < 300 and not ready():
        time.sleep(5)
    print("=== ready=%s 用时 %.0f s ===" % (ready(), time.time() - t0))
    if not ready():
        print("!!! 未就绪，请人工检查")
        return 1
    time.sleep(3)

    print("=== 复测（提示均唯一）===")
    for i in range(2):
        k = uuid.uuid4().hex[:6]
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 256,
                        "messages": [{"role": "user",
                                      "content": "Night watch log %s: describe the lighthouse in 200 words." % k}]})
        print("  decode %d tok in %.2fs = %.1f tok/s" % (r["usage"]["completion_tokens"], wall,
                                                         r["usage"]["completion_tokens"] / wall))
    k = uuid.uuid4().hex[:8]
    txt = ((PARA.replace("harbour", "harbour-" + k)) * 400) + "Question: one word - the fog colour? (ref %s)" % k
    r, wall = post({"model": "x", "temperature": 0, "max_tokens": 32,
                    "messages": [{"role": "user", "content": txt}]})
    pt = r["usage"]["prompt_tokens"]
    print("  prefill %d tok in %.2fs = %.0f tok/s" % (pt, wall, pt / wall))

    tail = subprocess.run(["grep", "-a", "serve: prompt", str(LOG)], capture_output=True,
                          text=True).stdout.splitlines()[-3:]
    print("=== 引擎自报 ===")
    for line in tail:
        print("  " + line[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
