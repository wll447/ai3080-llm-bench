#!/usr/bin/env python3
"""ai3080: 切换一个 Strata A/B 变体 -> 重启 -> 等 ready -> 跑干净基准 -> 打印结果。

干净基准的口径（吸取上一轮教训）：
  * 每个请求的提示内容都带唯一 nonce，避免 conversation cache 把 prefill 短路；
  * 长提示 ~31k token（超过 25088 的 chunk 上限，才测得到借用/分段那条路径）；
  * 重载后先热身，再取 3 次 prefill 与 2 次 decode，报中位数。

用法:
  strata_switch.py --label B --gpu 0,1,2 --unset-env STRATA_SPLIT_OWN --bench
  strata_switch.py --label A --gpu 0,1,2,3 --unset-env STRATA_SPLIT_OWN --bench
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import time
import urllib.request
import uuid

HOME = pathlib.Path.home()
CFG = HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.json"
LOG = HOME / "Strata" / "strata-unsloth-ud-q4_k_xl.log"
SESSION = "strata-262k"
API = "http://127.0.0.1:8080/v1/chat/completions"
PARA = ("The harbour lights were dim that evening, and the fog rolled in from the north-east in slow grey banks. "
        "Sailors say the beam must sweep the water twice before a boat may turn, and the keeper counts the turns aloud. "
        "Below the tower, the rocks are black and wet, and the tide makes a sound like breathing in a sleeping room. ")


def lines():
    try:
        return LOG.read_text(errors="replace").splitlines()
    except OSError:
        return []


def post(body, timeout=900):
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out, time.time() - t0


def bench():
    out = {"prefill": [], "decode": []}
    for i in range(3):
        n = uuid.uuid4().hex[:8]
        txt = ((PARA.replace("harbour", "harbour-" + n)) * 400) + \
              "Question: answer in one word - what colour is the fog described above? (ref %s)" % n
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 32,
                        "messages": [{"role": "user", "content": txt}]})
        pt = r["usage"]["prompt_tokens"]
        out["prefill"].append((pt, wall, pt / wall))
    for i in range(2):
        n = uuid.uuid4().hex[:8]
        r, wall = post({"model": "x", "temperature": 0, "max_tokens": 256,
                        "messages": [{"role": "user",
                                      "content": "Lighthouse keeper log entry %s: describe the night watch in 200 words." % n}]})
        ct = r["usage"]["completion_tokens"]
        out["decode"].append((ct, wall, ct / wall))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--gpu")
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--add", nargs=2, action="append", default=[])
    ap.add_argument("--env", nargs=2, action="append", default=[])
    ap.add_argument("--unset-env", action="append", default=[])
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    changed = []
    if args.gpu:
        cfg["gpu"] = [int(x) for x in args.gpu.split(",")]
        changed.append("gpu=%s" % cfg["gpu"])
    a = cfg["args"]
    for flag in args.drop:
        if flag in a:
            i = a.index(flag)
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                del a[i:i + 2]
            else:
                del a[i]
            changed.append("drop %s" % flag)
    for flag, val in args.add:
        if flag in a:
            a[a.index(flag) + 1] = val
        else:
            a += [flag, val]
        changed.append("add %s %s" % (flag, val))
    env = cfg.setdefault("env", {})
    for k, v in args.env:
        env[k] = v
        changed.append("env %s=%s" % (k, v))
    for k in args.unset_env:
        if env.pop(k, None) is not None:
            changed.append("unset env %s" % k)
    if not env:
        cfg.pop("env", None)
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    print("=== 变体 %s 改动: %s ===" % (args.label, "; ".join(changed) or "无"))
    print("=== gpu=%s env=%s ===" % (cfg["gpu"], cfg.get("env", {})))

    n0 = len(lines())
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
    print("=== 已重启，等待 ready ===")

    t0 = time.time()
    while time.time() - t0 < 600:
        if any("ready: http" in l for l in lines()[n0:]):
            break
        time.sleep(5)
    print("=== ready 用时 %.0f s ===" % (time.time() - t0))
    for line in lines()[n0:]:
        if any(k in line for k in ("prompt chunk auto", "layer split auto: K=", "prompts read slower",
                                   "expert cache auto", "other apps use the Chat settings")):
            print("  " + line[:190])

    if not args.bench:
        return 0
    time.sleep(3)
    r = bench()
    print("--- 基准（%s）---" % args.label)
    for pt, wall, rate in r["prefill"]:
        print("  prefill %d tok in %.2fs = %.0f tok/s" % (pt, wall, rate))
    print("  prefill 中位 = %.0f tok/s" % statistics.median(x[2] for x in r["prefill"]))
    for ct, wall, rate in r["decode"]:
        print("  decode %d tok in %.2fs = %.1f tok/s" % (ct, wall, rate))
    print("  decode 中位 = %.1f tok/s" % statistics.median(x[2] for x in r["decode"]))
    print("--- 引擎自报（最近 4 条）---")
    for line in lines()[-80:]:
        if "strata serve: prompt" in line:
            print("  " + line[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
