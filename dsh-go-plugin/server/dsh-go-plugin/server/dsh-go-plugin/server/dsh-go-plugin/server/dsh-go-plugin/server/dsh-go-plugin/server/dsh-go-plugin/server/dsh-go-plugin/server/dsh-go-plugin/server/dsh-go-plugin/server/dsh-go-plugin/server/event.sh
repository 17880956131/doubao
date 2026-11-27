#!/bin/sh
# 往数字人舞台推一个环境事件
# 用法：sh event.sh "外面下雨了" [类型]
# 类型默认 external；也可以写 low-battery / night / back / idle 等场景名
python3 - "$1" "${2:-external}" <<'PY'
import json, sys, urllib.request
text, kind = sys.argv[1], sys.argv[2]
body = json.dumps({"type": kind, "text": text}, ensure_ascii=False).encode("utf-8")
request = urllib.request.Request("http://127.0.0.1:8100/api/event", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
try:
    print(urllib.request.urlopen(request, timeout=5).read().decode())
except Exception as error:
    print("推送失败：", error)
PY
