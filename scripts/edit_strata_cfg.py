#!/usr/bin/env python3
"""ai3080: 给 Strata 的 UD-Q4_K_XL 运行配置改三个启动参数。
 --max-context            131072 -> 262144   (原生上限)
 --prefill                auto   -> auto:32768
 --conversation-cache-mib (新增) -> 8192
只改 args 里的这几项，其余原样保留；幂等，可重复执行。
"""
import json
import pathlib

p = pathlib.Path.home() / "Strata" / "strata-unsloth-ud-q4_k_xl.json"
cfg = json.loads(p.read_text(encoding="utf-8"))
a = cfg["args"]

WANTED = [("--max-context", "262144"),
          ("--prefill", "auto:32768"),
          ("--conversation-cache-mib", "8192")]

for name, value in WANTED:
    if name in a:
        a[a.index(name) + 1] = value
    else:
        a += [name, value]

p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
print(json.dumps(cfg, indent=1))
