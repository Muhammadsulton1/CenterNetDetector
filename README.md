# CenterNetDetector

Обучение детекции объектов с **голова в стиле CenterNet**: теплокарты по классам, регрессия размера (width/height) и смещения центра. В качестве **бэкбона** используется **DINOv2** из Hugging Face Transformers (`Dinov2Model`), признаки патч-токенов переформируются в карту признаков и доинтерполируются до шага `img_stride`.

## Стек и зависимости

Основные пакеты: PyTorch, torchvision, **transformers** (DINOv2), OpenCV, PyYAML, tqdm, torchmetrics.

Установка (из корня репозитория):

```bash
pip install -r requirements.txt
```

## Архитектура модели

| Компонент | Реализация |
|-----------|------------|
| **Бэкбон** | `Dinov2Model.from_pretrained(...)`, по умолчанию `facebook/dinov2-small` (см. [`model.py`](model.py)). Патчи **14×14** пикселей; в forward используются **все токены после первого** (`tok[:, 1:, :]`), сетка `H // 14 × W // 14`. |
| **Голова** | `CenterNetHead`: общий свёрточный ствол + три ветки — heatmap (sigmoid), wh (`softplus`), offset (линейная). Шаг карты признаков относительно входа задаётся `img_stride` (по умолчанию **4**, т.е. карта ~160×160 при `imgsz=640`). |
| **Лосс** | Focal-подобная функция для heatmap + L1 по wh и offset в точках центров объектов на карте (`model.py`). |

При создании модели у бэкбона выставляется `requires_grad=False` (**заморозка весов DINOv2**); в оптимизаторе параметры бэкбона всё равно перечислены, но без градиентов они **фактически не обновляются** — учится голова сетки.

Параграф секции `backbone` в [`configs/default.yaml`](configs/default.yaml) отражает желаемую конфигурацию, но **текущий `train.py` не читает эти поля**: имя модели DINOv2 задаётся в коде класса [`DinoV2CenterNet`](model.py) (аргумент `dino_name`), чтобы другой checkpoint подключить — правьте вызов в `train.py` / конструктор либо расширяйте загрузку из YAML.

## Формат данных

Проект ожидает датасет в **YOLO-подобном** виде и YAML с метаданными (как в экосистеме Ultralytics).

### Файл `dataset.yaml`

- **`path`** — корень набора данных (папка с `train`/`valid`/…).
- **`train`** / **`val`** — подпути к изображениям **относительно** `path` (например `train/images`, `valid/images`).
- **`nc`** — число классов.
- **`names`** — имена классов: список или словарь `{индекс: имя}`.

Если YAML с другой машины ссылается на чужой `path`, в учебном конфиге можно задать переопределение корня через `data.root_override` (локальная папка относительно корня проекта). См. [`configs/default.yaml`](configs/default.yaml) и функцию [`resolve_dataset`](train.py).

### Дерево папок (поддерживаются два варианта)

**Вариант 1:** split внутри корня

```
<root>/train/images/<name>.jpg
<root>/train/labels/<name>.txt
<root>/valid/images/...
<root>/valid/labels/...
```

Валидация в коде запрашивает split `valid`; в `dataset.yaml` часто поле называется `val:` — нужна папка `valid/` на диске или доработка кода под имя `val`.

**Вариант 2:** общие каталоги `images` / `labels`

```
<root>/images/train/<name>.jpg
<root>/labels/train/<name>.txt
```

То же для `valid`.

Изображения: расширения `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

### Файлы разметки `.txt`

Одна строка — один объект:

```
class_id cx cy w h
```

Все координаты и размеры **нормализованы** в [0, 1]: центр ограничивающего прямоугольника и ширина/высота относительно **исходного** изображения (до letterbox).

Пример демо-конфигурации: [`detection_dataset_yolo/dataset.yaml`](detection_dataset_yolo/dataset.yaml).

## Препроцессинг

- На вход модели после даталоадера изображение нормализуется **ImageNet** `mean` / `std` из конфига (см. `preprocess` в YAML); размер задаётся `preprocess.imgsz` (**letterbox** до квадрата, см. [`dataset.py`](dataset.py)).

## Обучение

Из корня проекта:

```bash
python train.py --config configs/default.yaml
```

Путь к `dataset.yaml` и опционально `root_override` берутся из секции **`data`** в том же конфиге. Устройство: `train.device` (при недоступном CUDA указанное `cuda` сбросится на CPU в коде).

Сохраняются чекпоинты под каталог `outputs.project` / `outputs.name` (по умолчанию `runs/exp`): **`best.pt`** (лучший по **mAP@0.50** на валидации) и **`last.pt`**.

В Checkpoint: `model`, `ema_model`, `optimizer`, `scheduler`, `epoch`, `metrics`, `names`.

EMA обновляется через `torch.optim.swa_utils.AveragedModel` в каждую эпоху валидации вызывается модель **`ema_model`**.

## Инференс

```bash
python inference.py --weights runs/exp/best.pt --source path/to/image_or_dir --out path/to/output --imgsz 640 --conf 0.1 --topk 100 --device cuda
```

Рисуются боксы на исходном изображении с учётом letterbox и отображения координат обратно в оригинальный размер.

## Конфигурация (`configs/default.yaml`)

| Секция | Назначение |
|--------|------------|
| `data` | `dataset_yaml`, `root_override` |
| `preprocess` | `imgsz`, `mean`, `std` |
| `augment` | в текущей версии **не подключено** к `train.py` (зарезервировано) |
| `backbone` | задумано под выбор модели; **пока только в YAML**, без автоподключения в коде |
| `model` | `img_stride`, bias heatmap, параметры focal / gaussian |
| `train` | эпохи, batch, LR, seed, device, EMA-связанные константы в скрипте |
| `metrics` | порог для eval, `topk_inference`, список IoU для torchmetrics MAP |
| `outputs` | куда писать `runs` и имя эксперимента |

## Структура репозитория (основное)

| Файл | Роль |
|------|------|
| `train.py` | цикл обучения, метрики MAP, сохранение чекпоинтов |
| `inference.py` | прогон по изображениям и визуализация |
| `model.py` | DINOv2 + CenterNetHead, loss, decode пиков |
| `dataset.py` | `YOLODataset`, letterbox, collate |
| `loss.py` | фокальный лосс для heatmap |
| `utils.py` | гауссианы для целевых карт |

## Лицензия

См. файл [`LICENSE`](LICENSE).
