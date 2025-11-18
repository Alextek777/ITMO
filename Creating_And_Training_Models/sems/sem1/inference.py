"""
inference.py — тестирование исходной и дообученной модели Mistral‑7B.

Скрипт позволяет выбрать, какую модель грузить: оригинальную (без адаптеров)
или дообученную (с LoRA‑адаптерами). Модель загружается в 4‑битном
квантизованном виде с помощью BitsAndBytesConfig, что позволяет экономить
память. После запуска скрипт принимает вопросы от пользователя и выводит
ответ модели. Для выхода введите пустую строку.

Запуск:

    python inference.py

Выберите режим: 1 — оригинальная модель, 2 — дообученная модель. Пути к
каталогам с весами можно переопределить через переменные ORIG_DIR и FT_DIR.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Пути к папкам с исходной и дообученной моделью. По умолчанию указывают
# на локальный диск C:\LLM. При необходимости измените.
ORIG_DIR = os.environ.get("LLM_ORIG", r"C:/LLM/mistral_model")
FT_DIR   = os.environ.get("LLM_FT", r"C:/LLM/ft_model")

def main() -> None:
    # Запрашиваем у пользователя выбор модели. 1 — оригинальная, 2 — дообученная.
    choice = input(
        "0 = выход\n1 = оригинальная модель\n2 = дообученная модель\n\nВыбор > "
    ).strip()
    if choice == "0":
        return
    elif choice == "1":
        model_dir = ORIG_DIR
    elif choice == "2":
        model_dir = FT_DIR
    else:
        raise SystemExit("Некорректный выбор, завершение.")

    # Настройки квантизации. load_in_4bit=True включает 4‑битный режим,
    # bnb_4bit_use_double_quant=True активирует двойную квантизацию, что
    # уменьшает потери точности, bnb_4bit_quant_type='nf4' задаёт нормальную
    # float4 схему, а bnb_4bit_compute_dtype=torch.bfloat16 сообщает, что
    # вычисления выполняются в bfloat16.
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Загружаем токенизатор. Если у модели нет pad_token, используем eos_token.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # Загружаем модель с LoRA‑адаптерами (если есть) в 4‑битном виде. device_map="auto"
    # распределяет модель по доступным GPU/CPU автоматически.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    gen_cfg = dict(max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print("Введите пустую строку для выхода.")
    while True:
        prompt = input("\nВаш вопрос > ").strip()
        if not prompt:
            break
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_cfg)
        print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()