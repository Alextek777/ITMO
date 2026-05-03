# EEG Motor Classification — Project Context

## Что это

Дипломный проект ИТМО. Детектор моторной активности мозга на основе EEG.
Задача: 8-классовая классификация моторных действий (кросс-субъект).

## Датасет

**MILimbEEG** (Mendeley v2)
- Путь: `/home/kalex/projects/ITMO/Diploma/datasets/MILimbEEG/MILimbEEG/`
- Устройство: OpenBCI Cyton + Daisy — 16 каналов, 125 Hz, аппаратный bandpass 5–50 Hz
- 60 испытуемых (S1–S60), только Motor файлы (M*.csv), 3721 триал
- Каждый файл: 500 сэмплов × 16 каналов = 4.0 секунды
- Формат имени: `S{subj}R{rep}M{label}_{trial}.csv`

**Классы (label → index):**
| Label | Index | Название | Триалов |
|-------|-------|----------|---------|
| M1 | 0 | BEO — Baseline Eyes Open | 60 |
| M2 | 1 | CLH — Closing Left Hand | 300 |
| M3 | 2 | CRH — Closing Right Hand | 300 |
| M4 | 3 | DLH — Dorsal Flexion Left Foot | 300 |
| M5 | 4 | PLF — Plantar Flexion Left Foot | 300 |
| M6 | 5 | DRF — Dorsal Flexion Right Foot | 300 |
| M7 | 6 | PRF — Plantar Flexion Right Foot | 300 |
| M8 | 7 | Rest | 1861 |

**Важно: сильный дисбаланс** — Rest в 31× больше BEO. Всегда используй `compute_class_weights()`.

**Электроды (CSV колонки 0–15 → CH_NAMES):**
```
FC5, F3, Fz, F4, FC6, FC3, FC4, Cz, T7, CP5, C3, CP3, CP4, C4, CP6, T8
```
Моторная кора: C3 (idx 10), Cz (idx 7), C4 (idx 13)

**Кросс-субъект сплит:**
- Train: S1–S48 (48 субъектов, ~80%)
- Val:   S49–S54 (6 субъектов, ~10%)
- Test:  S55–S60 (6 субъектов, ~10%)

## Структура проекта

```
detector/
├── CLAUDE.md               ← этот файл
├── config.py               ← все константы и пути (менять здесь)
├── train.py                ← CLI обучение (python train.py --model all)
├── requirements.txt
├── checkpoints/            ← сохранённые .keras модели
├── results/                ← метрики JSON, history CSV, картинки
├── notebooks/
│   ├── 01_data_exploration.ipynb   ← EDA + MNE визуализация
│   ├── 02_model_training.ipynb     ← обучение всех 3 моделей
│   └── 03_results_comparison.ipynb ← финальное сравнение
└── src/
    ├── data/
    │   ├── loader.py        ← load_dataset(), load_cross_subject_splits()
    │   └── preprocessing.py ← preprocess(), to_eegnet_format(), to_sequence_format(), augment()
    ├── models/
    │   ├── eegnet.py        ← build_eegnet(), get_compiled_eegnet()
    │   ├── resnet.py        ← build_resnet1d(), build_resnet1d_lite(), get_compiled_resnet()
    │   └── bilstm.py        ← build_bilstm(), build_bilstm_deep/lite(), get_compiled_bilstm()
    └── utils/
        ├── visualization.py ← plot_raw_trial, plot_channel_psd, plot_erp, plot_topomap_class,
        │                       plot_training_history, plot_confusion_matrix, plot_roc_curves,
        │                       plot_precision_recall, plot_model_comparison, plot_attention_weights
        └── metrics.py       ← evaluate(), per_class_metrics(), build_comparison_table()
```

## Форматы данных (КРИТИЧНО)

```python
# После load_dataset():
X  →  (N, 16, 500)   float32   # raw: (n_trials, channels, samples)

# После preprocessing:
preprocess(X)  →  (N, 16, 500)  # bandpass → CAR → z-score

# Для EEGNet (channels_last — единственный вариант на CPU):
to_eegnet_format(X)  →  (N, 16, 500, 1)  # height=C, width=T, depth=1

# Для ResNet1D и BiLSTM:
to_sequence_format(X)  →  (N, 500, 16)   # time-first
```

## Архитектуры

### EEGNet (~3 400 params)
- Вход: `(batch, 16, 500, 1)` — channels_last (CPU-совместимо, NCHW не работает на CPU в TF)
- Блок 1: Conv2D(F1=8, (1,64)) → BN → DepthwiseConv2D((16,1), D=2) → BN → ELU → AvgPool(1,4) → Dropout
- Блок 2: SeparableConv2D(F2=16, (1,16)) → BN → ELU → AvgPool(1,8) → Dropout
- Голова: Flatten → Dense(8, softmax)
- Компилируется через `get_compiled_eegnet()`

### ResNet1D (~665K params — Lite версия)
- Вход: `(batch, 500, 16)` — time-first
- Stem: Conv1D(64, 7, stride=2) → BN → ELU → MaxPool(3, stride=2)
- 3 стадии с bottleneck блоками (1×1 → K×1 → 1×1)
- GlobalAveragePooling1D → Dropout → Dense(8, softmax)
- Полный ResNet1D: 4 стадии, ~2M params
- Компилируется через `get_compiled_resnet(lite=True)`

### Bi-LSTM (~390K params)
- Вход: `(batch, 500, 16)` — time-first
- Проекция: Dense(64) → LayerNorm
- 2× Bidirectional(LSTM(units)) → LayerNorm → Dropout
- Temporal Attention (Bahdanau) → context vector
- Dense(64, elu) → Dropout → Dense(8, softmax)
- Варианты: `standard` [128,64], `deep` [256,128,64], `lite` [64]
- Компилируется через `get_compiled_bilstm(variant='standard')`

## Пайплайн обучения (train.py)

```python
# Типичный flow:
splits = load_cross_subject_splits()          # {train/val/test: (X, y, meta)}
X = preprocess(X_raw)                         # filter + CAR + zscore
X_eeg = to_eegnet_format(X)                  # для EEGNet
X_seq = to_sequence_format(X)                 # для ResNet/BiLSTM
cw = compute_class_weights(y_train)           # class weights dict
model.fit(..., class_weight=cw)
```

**Коллбэки:** ModelCheckpoint + EarlyStopping(patience=15) + ReduceLROnPlateau + CSVLogger + TensorBoard

## CLI быстрый запуск

```bash
# Обучить одну модель
python train.py --model eegnet
python train.py --model resnet --lite
python train.py --model bilstm --variant deep

# Все три сразу
python train.py --model all --epochs 80 --batch 64

# Без аугментации
python train.py --model eegnet --no-aug

# TensorBoard
tensorboard --logdir results/logs/
```

## Метрики

`evaluate(y_true, y_pred, y_prob)` возвращает:
- `accuracy`, `balanced_accuracy`
- `f1_macro`, `f1_weighted`
- `cohen_kappa` (κ — chance-corrected)
- `mcc` (Matthews Correlation Coefficient)
- `auc_macro`, `auc_weighted` (One-vs-Rest, нужен y_prob)

Сохраняются в `results/{ModelName}/metrics.json`

## Окружение и зависимости

- Python: 3.11 (system `/usr/bin/python3`)
- TensorFlow: 2.19 (установлен в `~/.local/lib/python3.11/`)
- MNE: 1.11 (установлен через `pip install mne --break-system-packages`)
- GPU: NVIDIA RTX 3060 (12 GB)
- **Проблема CUDA:** система имеет CUDA toolkit 11.2 (`nvcc`), но TF 2.19 собран под CUDA 12.x
  - Решение — установить CUDA 12.3 + cuDNN 9 через NVIDIA repo (Debian 12)
  - Временный обход: `CUDA_VISIBLE_DEVICES=""` для принудительного CPU
  - Без GPU проект работает, но медленнее

## Нерешённые задачи / что делать дальше

- [ ] Установить CUDA 12.3 + cuDNN 9 для GPU-ускорения
- [ ] Дообучить все 3 модели на полных данных (все 60 субъектов)
- [ ] Добавить LOSO (Leave-One-Subject-Out) кросс-валидацию
- [ ] Частотные фичи: STFT / Morlet wavelet вместо сырого сигнала
- [ ] Subject-specific fine-tuning (persona-адаптация)
- [ ] Ансамбль EEGNet + Bi-LSTM
- [ ] ShallowConvNet / DeepConvNet как baseline (Schirrmeister 2017)
- [ ] Заполнить таблицу в Notebook 03 реальными результатами обучения

## Ключевые наблюдения по данным

- μ-диапазон (8–13 Hz): контралатеральная ERD отличает руку vs ногу
- C3/C4: реагируют на движения рук; Cz — на ноги
- BEO (60 триалов) — практически невозможно обучить надёжно; рассмотреть объединение с Rest
- Preprocessing: bandpass [1–49 Hz] → CAR → z-score (в таком порядке)
- Аугментация только на train: Gaussian noise (std=0.05) + time-shift + channel dropout
