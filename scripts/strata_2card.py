#!/usr/bin/env python3
"""ai3080: 实测双卡（gpu=[0,1]）性能，测完自动恢复三卡 gpu=[0,1,2]。

口径与前面一致：提示内容唯一（带 nonce，防 conversation cache 短路）；
就绪判定用 HTTP /health（server.py 的 ready 横幅只进 tmux 面板，不进日志）。
"""
import json
import pathlib
import shutil
import statistics
import subprocess
import time
import urllib.request
import uuid

HOME = pathlib.Path.home()
STR = HOME / "Strata"
CFG = STR / "strata-unsloth-ud-q4_k_xl.json"
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
        return json.load(resp), time.time() - t0


def lines():
    try:
        return LOG.read_text(errors="replace").splitlines()
    except OSError:
        return []


def set_gpu(cards):
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg["gpu"] = list(range(cards))
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def restart(cards):
    n0 = len(lines())
    set_gpu(cards)
    print("=== gpu=%s 重启 ===" % list(range(cards)))
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
                    "cd %s && exec ./run-unsloth-ud-q4_k_xl.sh" % STR], check=False)
    t0 = time.time()
    while time.time() - t0 < 300 and not ready():
        time.sleep(5)
    ok = ready()
    print("=== ready=%s 用时 %.0f s ===" % (ok, time.time() - t0))
    time.sleep(2)
    for line in lines()[n0:]:
        if any(k in line for k in ("layer split auto: K=", "expert cache", "prompt chunk auto",
                                   "prompts read slower", "loaded ")):
            print("  " + line[:190])
    pane = subprocess.run(["tmux", "capture-pane", "-p", "-S", "-400", "-t", SESSION],
                          capture_output=True, text=True).stdout.splitlines()
    for line in pane:
        if "SPLIT_OWN" in line or "experts loaded" in line:
            print("  [pane] " + line[:170])
    return ok


def bench(label):
    print("--- 基准 %s ---" % label)
    rates = []
    for _ in range(3):
        k = uuid.uuid4().hex[:8]
        txt = ((PARA.replace("harbour", "harbour-" + k)) * 400) + \
              "Question: one word - the fog colour? (ref %s)" % k
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 32,
                        "messages": [{"role": "user", "content": txt}]})
        pt = r["usage"]["prompt_tokens"]
        rates.append(pt / wall)
        print("  prefill %d tok in %.2fs = %.0f tok/s" % (pt, wall, pt / wall))
    print("  prefill 中位 = %.0f tok/s" % statistics.median(rates))
    drates = []
    for _ in range(2):
        k = uuid.uuid4().hex[:6]
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 256,
                        "messages": [{"role": "user",
                                      "content": "Night watch log %s: describe the lighthouse in 200 words." % k}]})
        ct = r["usage"]["completion_tokens"]
        drates.append(ct / wall)
        print("  decode %d tok in %.2fs = %.1f tok/s" % (ct, wall, ct / wall))
    print("  decode 中位 = %.1f tok/s" % statistics.median(drates))


def main():
    shutil.copy2(CFG, STR / "strata-unsloth-ud-q4_k_xl.json.bak-before-2card")
    print("=== 已备份配置 ===")
    if not restart(2):
        print("!!! 双卡没起来，直接恢复三卡")
    else:
        time.sleep(3)
        bench("2 卡")
        tail = subprocess.run(["grep", "-a", "serve: prompt", str(LOG)], capture_output=True,
                              text=True).stdout.splitlines()[-6:]
        print("=== 引擎自报（双卡）===")
        for line in tail:
            print("  " + line[:190])
    print("=== 恢复三卡 ===")
    restart(3)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
