#!/usr/bin/env python3
"""ai3080: 落地推荐配置（3 卡 + STRATA_SPLIT_OWN=1 + prefill auto）并测单流与 4 路并发。

要点：readiness 必须用 HTTP /health 判断 —— server.py 的 `ready:` 横幅只打印到 tmux 面板，
不写进 strata-<model>.log（这是上一轮白等 600 秒的原因）。
"""
import json
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid

HOME = pathlib.Path.home()
CFG = HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.json"
SESSION = "strata-262k"
API = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH = "http://<LAN-IP>:8080/health"
PARA = ("The harbour lights were dim that evening, and the fog rolled in from the north-east in slow grey banks. "
        "Sailors say the beam must sweep the water twice before a boat may turn, and the keeper counts the turns aloud. "
        "Below the tower, the rocks are black and wet, and the tide makes a sound like breathing in a sleeping room. ")


def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def ready():
    try:
        return '"loaded": true' in http_get(HEALTH)
    except Exception:
        return False


def post(body, timeout=900):
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, time.time() - t0


def prefill_runs(n=3):
    rates = []
    for _ in range(n):
        k = uuid.uuid4().hex[:8]
        txt = ((PARA.replace("harbour", "harbour-" + k)) * 400) + \
              "Question: answer in one word - what colour is the fog described above? (ref %s)" % k
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 32,
                        "messages": [{"role": "user", "content": txt}]})
        pt = r["usage"]["prompt_tokens"]
        rates.append(pt / wall)
        print("  prefill %d tok in %.2fs = %.0f tok/s" % (pt, wall, pt / wall))
    print("  prefill 中位 = %.0f tok/s" % statistics.median(rates))


def decode_runs(n=2):
    rates = []
    for _ in range(n):
        k = uuid.uuid4().hex[:6]
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 256,
                        "messages": [{"role": "user",
                                      "content": "Lighthouse keeper log %s: describe the night watch in 200 words." % k}]})
        ct = r["usage"]["completion_tokens"]
        rates.append(ct / wall)
        print("  decode %d tok in %.2fs = %.1f tok/s" % (ct, wall, ct / wall))
    print("  decode 中位 = %.1f tok/s" % statistics.median(rates))


def concurrent(n, tokens=200):
    """n 路同时发（唯一提示），返回 (总输出 token, 墙钟, 聚合 tok/s, 每路 tok/s)。"""
    out = {}
    lock = threading.Lock()

    def one(i):
        k = uuid.uuid4().hex[:6]
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": tokens,
                        "messages": [{"role": "user",
                                      "content": "Lighthouse log %s-%d: describe the night watch in 150 words." % (k, i)}]})
        with lock:
            out[i] = (r["usage"]["completion_tokens"], wall)

    t0 = time.time()
    ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    total = sum(v[0] for v in out.values())
    per = [v[0] / v[1] for v in out.values()]
    print("  并发 %d 路: 总输出 %d tok / 墙钟 %.2fs = 聚合 %.1f tok/s；单路 %.1f-%.1f tok/s"
          % (n, total, wall, total / wall, min(per), max(per)))


def main():
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg["gpu"] = [0, 1, 2]
    a = cfg["args"]
    if "--prefill" in a:
        a[a.index("--prefill") + 1] = "auto"
    cfg.setdefault("env", {})["STRATA_SPLIT_OWN"] = "1"
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("=== 落地推荐配置: gpu=%s env=%s prefill=%s ===" %
          (cfg["gpu"], cfg["env"], a[a.index("--prefill") + 1]))

    for pat in ("serve/server.py", "engine/strata"):
        subprocess.run(["pkill", "-INT", "-f", pat], check=False)
    for _ in range(20):
        if subprocess.run(["pgrep", "-f", "serve/server.py"], capture_output=True).returncode != 0:
            break
        time.sleep(2)
    subprocess.run(["pkill", "-TERM", "-f", "serve/server.py"], check=False)
    subprocess.run(["pkill", "-TERM", "-f", "engine/strata"], check=False)
    time.sleep(3)
    subprocess.run(["tmux", "kill-session", "-t", SESSION], check=False)
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION,
                    "cd %s/Strata && exec ./run-unsloth-ud-q4_k_xl.sh" % HOME], check=False)

    t0 = time.time()
    while time.time() - t0 < 300 and not ready():
        time.sleep(5)
    print("=== ready=%s 用时 %.0f s（HTTP 探测）===" % (ready(), time.time() - t0))
    if not ready():
        return 1
    time.sleep(3)

    print("--- 单流 ---")
    prefill_runs()
    decode_runs()
    print("--- 并发 ---")
    concurrent(1)
    concurrent(4)
    print("--- 尾部配置确认 ---")
    print("  layer split:", [l for l in subprocess.run(
        ["tail", "-200", str(HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.log")],
        capture_output=True, text=True).stdout.splitlines() if "layer split auto: K=" in l][-1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
