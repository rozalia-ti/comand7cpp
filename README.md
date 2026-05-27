# Командная работа 7. Классификация через C++ модель + Python-обертка

## Что делает проект

Проект реализует нейросеть на C++ для бинарной классификации и предоставляет к ней Python API через `pybind11`.

По текущей версии лабораторной задачи:
- загружаются CSV-датасеты (произвольное число признаков + метка)
- производится разделение 80/20 (train/val)
- обучается модель в Python-скрипте `train_classifier.py`
- считается `F1` на каждом датасете
- итоговая оценка считается по формуле
  `0.5 * F1(d1) + 0.5 * F1(d2)`
- есть возможность дообучения (fine-tune) на третьем датасете

## Что где находится

- `src/wrapper.cpp` — Python-обертка (`pybind11`) класса `Model`.
  Экспортируется в модуль `comand7`.
- `include/ai.h` — реализация нейронной сети (`AiMatrix`) с прямым и обратным распространением.
- `src/main.cpp` — демонстрационный C++ сценарий (визуализация и эксперимент на синтетике).
- `train_classifier.py` — основной запуск обучения/оценки по ТЗ.
- `Makefile` — сборка `.so`/`.pyd` модуля.
- `include/distribution.h` — генераторы данных (используются в основном C++-классе, в текущем пайплайне не являются основным источником данных).
- `include/point.h`, `src/point.cpp` — структура точки.
- `include/display.h`, `src/display.cpp`, `src/display_platform/*` — консольная отрисовка (поддержка отладки и визуального режима).
- `dataset1.csv` — пример первого датасета (в репозитории лежит только один датасет, второй нужно добавить в проект вручную).
- `model.bin` — путь по умолчанию для сохранения весов модели после тренировки.
- `comand7<suffix>` — уже собранный Windows-модуль, если он есть в окружении.

## Функции и API

### Класс `Model` в модуле `comand7`

Конструкторы:
- `Model()` — инициализация в режиме аппроксимации линии (legacy).
- `Model(in_size, hidden1=32, hidden2=16)` — инициализация классификатора.

Методы для обучения/предсказания классификации:
- `fit(x_batch: list[list[float]], y_batch: list[float], lr: float)`
- `predict_proba(sample: list[float]) -> float`
- `predict(sample: list[float]) -> int`
- `predict_batch(x_batch) -> list[int]`
- `predict_proba_batch(x_batch) -> list[float]`
- `save(filename: str)` / `load(filename: str)` — сохранение/загрузка весов (`AiMatrix::save/load`).

Старые методы (line-mode, для совместимости):
- `train_arbitraty(batches, epochs)`
- `predict_line(raw_points)`
- `predict_point_class(x, y)`

### Как работает модель внутри

- `AiMatrix` хранит слои (`AiLayer`) с нейронами, весами и дельтами.
- Имеется настройка архитектуры через `configure(...)`, случайная инициализация `randomize()`.
- Обучение: `train(input, target, lr)` делает forward + backward + обновление весов.
- В классификационном режиме выходной нейрон — sigmoid, порог отсечки — `0.5`.

## Механизмы и подходы

- **Сеть**: многослойный перцептрон с ручной реализацией прямого/обратного прохода.
- **Активации**: `ReLU` в скрытых слоях, `Sigmoid` на выходе для бинарной классификации.
- **Экспорт в Python**: `pybind11`.
- **Метод оценки**: `Precision`, `Recall`, `F1` по формуле `2*P*R/(P+R)`.
- **Валидация**: фиксированный `80/20` раздел (`train_test_split`).
- **Нормализация данных**: z-score по тренировочному `d1`.

## Как запустить (Windows)
### 1) Подготовка окружения (рекомендуется)

```powershell
cd path\to\comand7cpp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pybind11 matplotlib
```

Если PowerShell блокирует активацию скрипта:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2) Сборка Python-модуля

```powershell
make
```

После сборки модуль появляется в `build\bin` (для Windows это `.pyd`).

### 3) Обучение и оценка

```powershell
python .\train_classifier.py --d1 dataset1.csv --d2 dataset2.csv
```

Второй датасет обязателен по ТЗ; если не указан, используется `dataset1.csv`.

По умолчанию скрипт делает `d2=dataset1.csv`, чтобы можно было быстро запустить `--epochs 1` и проверить пайплайн.

Скрипт:
- делит `d1` и `d2` по `80/20` (по умолчанию `--split 0.2`),
- обучает модель на `d1`,
- выводит `F1_d1`, `F1_d2`, итоговый балл и статус `Accepted`/`Not accepted`.

### 4) Дообучение на третьем датасете

```powershell
python .\train_classifier.py --d1 dataset1.csv --d2 dataset2.csv --finetune dataset3.csv
```

### 5) Сохранение весов

По умолчанию:

```powershell
--model model.bin
```

Можно задать путь явно через `--model`.

### 6) Полезные команды

```powershell
make clean          # удалить build
make rebuild        # чистая пересборка
python -m py_compile train_classifier.py   # синтакс. проверка скрипта
```

### 7) Прогон без графика (стандартный)

```powershell
python .\train_classifier.py --d1 dataset1.csv --d2 dataset2.csv --epochs 80 --eval-every 1
```

### 8) Запуск на другом ноутбуке (чистый старт)

```powershell
git clone <URL_репозитория> comand7cpp
cd comand7cpp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pybind11 matplotlib
git fetch origin
git checkout defence
git pull
make clean
make
python .\train_classifier.py --d1 dataset1.csv --d2 dataset2.csv --epochs 80 --eval-every 1
```

```powershell
python -c "import sys; sys.path.insert(0, 'build\\bin'); import comand7; print('module:', comand7.__file__)"
```

## Аргументы скрипта `train_classifier.py`

- `--d1` (по умолчанию `dataset1.csv`) — путь к первому датасету.
- `--d2` (по умолчанию `dataset1.csv`) — путь ко второму.
- `--split` (по умолчанию `0.2`) — доля теста.
- `--epochs` (по умолчанию `200`) — эпох на `d1`.
- `--batch-size` (по умолчанию `16`).
- `--lr` (по умолчанию `0.005`).
- `--hidden1`, `--hidden2` (по умолчанию `64`, `32`) — размеры скрытых слоев.
- `--seed` (по умолчанию `42`).
- `--model` — путь сохранения модели.
- `--load` — загрузить существующие веса и продолжить.
- `--finetune` — путь к третьему датасету.
- `--finetune-epochs` — эпох дообучения (по умолчанию `50`).
- `--eval-every` — период печати промежуточного F1 по `d1`.

## Формат входного CSV

- Заголовок обязателен.
- Должно быть хотя бы 1 признака и одна колонка цели.
- Колонку цели скрипт определяет по именам: `target`, `y`, `label`, `class`, `cls`.

Пример структуры:

```csv
feature_0,feature_1,feature_2,target
1.2,-0.7,0.5,0
0.4,2.3,-1.2,1
```

## Формат выходного/внутреннего представления

- Признаки в модели нормализуются к виду z-score перед обучением и инференсом.
- Предсказание класса: `>= 0.5 -> 1`, иначе `0`.

## Важные замечания

- Для обучения по ТЗ используйте не `train_arbitraty`, а `train_classifier.py`.
- Модуль `Model()` без параметров — legacy режим линейной аппроксимации; для классификации всегда используйте `Model(in_size, hidden1, hidden2)`.
- Сборка зависит от доступности `pybind11` в используемой версии `python`.
