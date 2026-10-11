#!/usr/bin/env python3
"""ai3080: 真并发对比 —— 3 卡 vs 4 卡，在 N=1/2/4 路同时请求下的聚合吞吐。

前提：必须开引擎的并行解码（--parallel N），否则请求只是排队（上一轮踩过的坑）。
每个请求的提示都带唯一 nonce，绝不重复（用户明确要求）。
每个配置跑完打印：聚合 tok/s、单路 tok/s 区间、以及相对 N=1 的放大倍数。
"""
import json
import pathlib
import subprocess
import threading
import time
import urllib.request
import uuid

HOME = pathlib.Path.home()
CFG = HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.json"
SESSION = "strata-262k"
API = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH = "http://127.0.0.1:8080/health"
TOKENS = 256
PARALLEL = "4"


def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def ready():
    try:
        return '"loaded": true' in http_get(HEALTH)
    except Exception:
        return False


def post(body, timeout=1200):
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, time.time() - t0


def burst(n):
    """n 路同时发（每路唯一提示），返回 (总输出 token, 墙钟 s, 单路 tok/s 列表)。"""
    res = {}
    lock = threading.Lock()
    barrier = threading.Barrier(n)

    def one(i):
        k = uuid.uuid4().hex[:8]
        body = {"model": "x", "temperature": 0, "max_tokens": TOKENS,
                "messages": [{"role": "user",
                              "content": "Watch log %s entry %d: describe a storm at the lighthouse in about 200 words." % (k, i)}]}
        barrier.wait()
        try:
            r, wall = post(body)
            got = (r["usage"]["completion_tokens"], wall)
        except Exception as e:
            got = (0, 1.0)
            print("    request %d failed: %s" % (i, e))
        with lock:
            res[i] = got

    ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    t0 = time.time()
    for t in ths:
        t.join()
    wall = time.time() - t0
    total = sum(v[0] for v in res.values())
    per = [v[0] / v[1] for v in res.values()]
    return total, wall, per


def set_cfg(cards, parallel):
    """parallel 是运行配置的顶层键 cfg["parallel"]（server.py 的 parallel_args() 会翻成引擎 --batch），
    不是引擎参数、也不是命令行旗标 —— 绝不能塞进 args。"""
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg["gpu"] = list(range(cards))
    a = cfg["args"]
    if "--prefill" in a:
        a[a.index("--prefill") + 1] = "auto"
    if parallel:
        cfg["parallel"] = int(parallel)
    else:
        cfg.pop("parallel", None)
    cfg.setdefault("env", {})["STRATA_SPLIT_OWN"] = "1"
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


def restart(cards, parallel):
    cfg = set_cfg(cards, parallel)
    print("=== 配置: gpu=%s parallel=%s ===" % (cfg["gpu"], parallel))
    for pat in ("[s]erve/server.py", "[e]ngine/strata"):
        subprocess.run(["pkill", "-INT", "-f", pat], check=False)
    for _ in range(20):
        if subprocess.run(["pgrep", "-f", "[s]erve/server.py"], capture_output=True).returncode != 0:
            break
        time.sleep(2)
    subprocess.run(["pkill", "-TERM", "-f", "[s]erve/server.py"], check=False)
    subprocess.run(["pkill", "-TERM", "-f", "[e]ngine/strata"], check=False)
    time.sleep(3)
    subprocess.run(["tmux", "kill-session", "-t", SESSION], check=False)
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION,
                    "cd %s/Strata && exec ./run-unsloth-ud-q4_k_xl.sh" % HOME], check=False)
    t0 = time.time()
    while time.time() - t0 < 300 and not ready():
        time.sleep(5)
    ok = ready()
    print("=== ready=%s 用时 %.0f s ===" % (ok, time.time() - t0))
    if not ok:
        return False
    time.sleep(3)
    tail = subprocess.run(["tail", "-250", str(HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.log")],
                          capture_output=True, text=True).stdout.splitlines()
    for line in tail:
        if "slots are live" in line or "prompt chunk auto" in line or "layer split auto: K=" in line:
            print("  [log] " + line[:180])
    # server.py 的 [strata] 消息只进 tmux 面板，不进日志
    pane = subprocess.run(["tmux", "capture-pane", "-p", "-S", "-400", "-t", SESSION],
                          capture_output=True, text=True).stdout.splitlines()
    for line in pane:
        if any(k in line for k in ("parallel", "batch", "slots", "ready:", "experts loaded")):
            print("  [pane] " + line[:180])
    return True


def main():
    print("### 目标：3 卡 vs 4 卡，在 N=1/2/4 路真并发下的聚合吞吐（每路提示唯一）###")
    for cards in (3, 4):
        if not restart(cards, PARALLEL):
            print("!!! %d 卡启动失败，跳过" % cards)
            continue
        base = None
        for n in (1, 2, 4):
            total, wall, per = burst(n)
            agg = total / wall
            if n == 1:
                base = agg
            print("  [%d 卡] N=%d: 总 %d tok / %.2fs = 聚合 %.1f tok/s；单路 %.1f-%.1f；相对 N=1 ×%.2f"
                  % (cards, n, total, wall, agg, min(per), max(per), agg / base))
            time.sleep(2)
    print("=== 收尾：恢复推荐配置（3 卡 + SPLIT_OWN，关闭 parallel）===")
    restart(3, None)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
