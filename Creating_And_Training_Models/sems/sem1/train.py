"""
train.py — дообучение Mistral‑7B с QLoRA (адаптирован под Transformers v4/v5).

Запуск:  python train.py

Все пути и шаги зашиты ниже. Никаких переменных окружения не нужно.
"""

from __future__ import annotations
import math
import os
from typing import Dict, Any

import torch
from datasets import load_from_disk
from packaging.version import Version
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
try:
    from transformers import BitsAndBytesConfig  # v4.30+ / v5
except Exception:
    BitsAndBytesConfig = None  # на очень старых версиях не будет; но тогда 4-bit недоступен

# ──────────────────────────────────────────────────────────────────────────────
# ЖЁСТКИЕ НАСТРОЙКИ — ПОД СВОИ ПАПКИ (Windows-пути поддерживаются)
# ──────────────────────────────────────────────────────────────────────────────
MAX_STEPS: int = 40  # быстрый прогон; поставьте -1 для эпохи
MODEL_DIR: str = r"C:/LLM/mistral_model"  # локальные веса и конфиги (включая .safetensors)
DS_PATH:   str = r"C:/LLM/my_ds"          # датасет, сохранённый load_from_disk(...)
OUT_DIR:   str = r"C:/LLM/ft_model"       # сюда положим LoRA-адаптеры и токенизатор

# Токенизация / обучение
MAX_SEQ_LEN: int = 512
LEARNING_RATE: float = 2e-4
GRAD_ACCUM_STEPS: int = 16
BATCH_SIZE_PER_DEVICE: int = 1
EVAL_STEPS: int = 50
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 200
BF16: bool = True
USE_GRADIENT_CHECKPOINTING: bool = True

HF_VERSION = Version(transformers.__version__.split('+')[0])
IS_V5 = HF_VERSION.major >= 5

def _build_model_kwargs() -> Dict[str, Any]:
    """Готовим параметры загрузки модели с учётом версии Transformers."""
    kwargs: Dict[str, Any] = {}

    # Настраиваем 4‑битную квантизацию через BitsAndBytes (без устаревших флагов)
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "В вашей версии transformers нет BitsAndBytesConfig. "
            "Обновите transformers / установите bitsandbytes, или отключите 4‑битный режим."
        )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    kwargs["quantization_config"] = bnb_config
    kwargs["device_map"] = "auto"

    # Параметр dtype переименован в v5 (вместо torch_dtype)
    if IS_V5:
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    return kwargs

def _make_training_args() -> TrainingArguments:
    """Создаём TrainingArguments, совместимые и с v4, и с v5."""
    base_kwargs: Dict[str, Any] = dict(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        max_steps=MAX_STEPS if MAX_STEPS and MAX_STEPS > 0 else None,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        optim="paged_adamw_32bit",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        disable_tqdm=False,
    )

    # Стратегии сохранения/оценки слегка переименовали в v5
    if IS_V5:
        base_kwargs["eval_strategy"] = "steps"
    else:
        base_kwargs["evaluation_strategy"] = "steps"
    base_kwargs["eval_steps"] = EVAL_STEPS

    # Остальные флаги обычно одинаковые
    base_kwargs["bf16"] = BF16
    base_kwargs["gradient_checkpointing"] = USE_GRADIENT_CHECKPOINTING

    # Последняя попытка «подстроиться»: если какая-то опция неожиданно не поддерживается,
    # создадим объект поэтапно.
    try:
        return TrainingArguments(**base_kwargs)
    except TypeError as e:
        msg = str(e)
        # Пробуем заменить ключи, если библиотека резко новее/старее
        if "evaluation_strategy" in msg:
            base_kwargs.pop("evaluation_strategy", None)
            base_kwargs["eval_strategy"] = "steps"
        if "eval_strategy" in msg:
            base_kwargs.pop("eval_strategy", None)
            base_kwargs["evaluation_strategy"] = "steps"
        if "gradient_checkpointing" in msg:
            base_kwargs.pop("gradient_checkpointing", None)
        if "bf16" in msg:
            base_kwargs.pop("bf16", None)
        if "optim" in msg:
            # На случай если оптимизатор переименован — убираем, Trainer сам подберёт дефолт
            base_kwargs.pop("optim", None)
        return TrainingArguments(**base_kwargs)

class MetricsPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        to_print = {k: float(v) if isinstance(v, (int, float)) else v for k, v in logs.items()}
        # Память GPU (если есть)
        try:
            if torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
                to_print["gpu_mem_mb"] = round(mem_mb, 2)
        except Exception:
            pass
        print(f"Шаг {state.global_step}: {to_print}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        eval_loss = float(metrics["eval_loss"])
        ppl = math.exp(eval_loss) if eval_loss < 100 else float('inf')
        print(f"\nОценка: step={state.global_step}, eval_loss={eval_loss:.4f}, perplexity={ppl:.4f}\n")

def main() -> None:
    # 1) Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tok(example):
        enc = tokenizer(
            f"{example.get('instruction','')}\n{example.get('input','')}\nОтвет:\n{example.get('output','')}",
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # 2) Датасет
    dataset = load_from_disk(DS_PATH)
    dataset_tok = dataset.map(tok, remove_columns=dataset["train"].column_names)

    # 3) Модель (4‑бит bnb, совместимо с v4/v5 API)
    model_kwargs = _build_model_kwargs()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **model_kwargs)
    model = prepare_model_for_kbit_training(model)

    # 4) LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 5) Аргументы обучения (совместимые)
    train_args = _make_training_args()

    # 6) Тренер
    eval_split = "test" if "test" in dataset_tok else None
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset_tok["train"],
        eval_dataset=dataset_tok[eval_split] if eval_split else None,
        callbacks=[MetricsPrinterCallback()],
    )

    # 7) Обучение
    trainer.train()

    # 8) Сохранение
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Готово. Адаптеры/токенизатор сохранены в: {OUT_DIR}")

if __name__ == "__main__":
    main()
