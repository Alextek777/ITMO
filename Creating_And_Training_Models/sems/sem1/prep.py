"""
prep.py — подготовка обучающего датасета для fine‑tuning.

Скрипт читает исходный файл с примерами (CSV или JSONL), преобразует его
в формат HuggingFace Dataset, делит данные на обучающую и валидационную части
и сохраняет результат на диск. Формат Arrow обеспечивает эффективную
загрузку примеров во время обучения без перегрузки оперативной памяти.

Чтобы запустить скрипт, установите зависимости `pandas` и `datasets` и
выполните команду:

    python prep.py

По умолчанию скрипт ищет исходный CSV в каталоге «C:/LLMtrain-data.csv»
и сохраняет готовый набор данных в «C:/LLM/my_ds». При желании вы
можете переопределить пути, изменив переменные `csv_path` и `out_path` ниже.
"""

import os
from datasets import Dataset
import pandas as pd

# Путь к исходному файлу с обучающими примерами. Формат CSV с колонками
# instruction, input и output. При необходимости замените на свой файл.
csv_path = os.environ.get("LLM_TRAIN_CSV", r"C:/LLM/train-data.csv")
# Папка для сохранения готового датасета в формате Arrow. Будет создана,
# если не существует.
out_path = os.environ.get("LLM_DS_OUT", r"C:/LLM/my_ds")

def main() -> None:
    """Читает CSV, делит на train/test и сохраняет."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Не найден исходный CSV: {csv_path}. Укажите путь в переменной LLM_TRAIN_CSV."
        )
    # Загружаем таблицу в DataFrame. keep_default_na=False оставляет пустые строки
    # как пустые строки, не преобразуя их в NaN.
    df = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
    if not {'instruction', 'input', 'output'}.issubset(df.columns):
        raise ValueError("CSV должен содержать колонки instruction, input и output")
    # Создаём объект Dataset из pandas. preserve_index=False — сохраняем только
    # данные, без индекса.
    dataset = Dataset.from_pandas(df, preserve_index=False)
    # Делим набор данных: 95 % в обучающую часть, 5 % — валидация. Это позволяет
    # отслеживать переобучение во время тренировки.
    dataset = dataset.train_test_split(test_size=0.05)
    # Создаём папку, если её ещё нет, и сохраняем набор данных на диск.
    os.makedirs(out_path, exist_ok=True)
    dataset.save_to_disk(out_path)
    print(f"Датасет сохранён в {out_path}")

if __name__ == "__main__":
    main()