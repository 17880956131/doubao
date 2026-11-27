# B 路线：训练数据与蒸馏

## 1. 导出数据（手机可做）

```sh
python3 export_sft.py
# 输出 sft.jsonl：system(自我内核) + user + assistant
```

## 2. 训练（需要 GPU，手机只做推理）

在电脑 / Colab / Kaggle / 云 GPU 上：

```sh
pip install torch transformers datasets peft trl bitsandbytes
python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data sft.jsonl --out zhiming-lora
```

- 1.5B QLoRA：T4/P100 级别即可，几百到几千条数据约 10~30 分钟
- 3B 也行，显存建议 ≥8GB

## 3. 合并 → GGUF → 手机

```sh
python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "zhiming-lora").merge_and_unload()
model.save_pretrained("zhiming-merged"); AutoTokenizer.from_pretrained("zhiming-lora").save_pretrained("zhiming-merged")
PY
# 用 llama.cpp 转换并量化
python convert_hf_to_gguf.py zhiming-merged --outtype q4_k_m --outfile zhiming-1.5b-q4.gguf
```

把 gguf 放到 `/sdcard/模型文件/`，改 `local-brain/start-llm.sh` 的模型路径，重启 8181。

## 4. 手机端的限制

- proot 里拿不到 GPU（没有 `/dev/kgsl`、`/dev/mali`、`/dev/dri`）
- 手机可以：数据采集、推理、导出、评估
- 手机不能：QLoRA 训练
