# CenterNetDetector

Обучение детекции объектов с **голова в стиле CenterNet**: теплокарты по классам, регрессия размера (width/height) и смещения центра. В качестве **бэкбона** используется **DINOv2** из Hugging Face Transformers (`Dinov2Model`), признаки патч-токенов переформируются в карту признаков и доинтерполируются до шага `img_stride`.

## Стек и зависимости

Основные пакеты: PyTorch, torchvision, **transformers** (DINOv2), OpenCV, PyYAML, tqdm, torchmetrics.

Установка (из корня репозитория):

```bash
pip install -e .
```

## Архитектура модели

| Компонент | Реализация |
|-----------|------------|
| **Бэкбон** | Любой из [`centernet/backbone.py`](centernet/backbone.py), выбирается ключом `model=`. **ViT-семейство** (HuggingFace): `facebook/dinov2-small`, `facebook/dinov2-with-registers-small`, DINOv3-ViT (`facebook/dinov3-vits16-pretrain-lvd1689m`, веса требуют `huggingface-cli login` и принятия лицензии), обычный `google/vit-*`. Размер патча (14 или 16) и число регистр-токенов берутся из config, CLS и регистры в карту признаков не попадают. **Иерархические** (HuggingFace `AutoBackbone` и `timm/<имя>`): ConvNeXt, DINOv3-ConvNeXt, ResNet и т. п., карты шагов 4-32 сливаются простым FPN. |
| **Признаки** | ViT: патч-токены `feature_layers` равномерно выбранных слоёв (по умолчанию 4, включая последний), склеенные по каналам. Картинка дополняется до кратного патчу размера, карта билинейно доводится до шага `img_stride` (по умолчанию 4) и обрезается: ячейка карты точно соответствует своим пикселям. |
| **Голова** | `CenterNetHead`: общий свёрточный ствол + три ветки: heatmap (sigmoid), wh (`softplus`), offset. |
| **Лосс** | Focal loss для heatmap + L1 по wh (вес 0.1) и offset в центрах объектов. Сумма по батчу делится один раз на число объектов в батче ([`model.py`](centernet/model.py)). |

Предобученная часть бэкбона по умолчанию **заморожена** (учатся голова и neck). `freeze_epochs=N` обучает только голову N эпох, затем размораживает бэкбон с `lr_backbone` (`freeze_epochs=0` - сразу, `-1` - никогда). Нормализация входа ImageNet mean/std (подходит DINOv2/DINOv3/ConvNeXt/большинству timm CNN; для timm-ViT с другой нормализацией её нужно менять).

Проверка на разных бэкбонах: `tests/test_model.py` гоняет forward/backward/infer на маленьких случайных моделях каждого семейства (без скачивания). Реальные веса из кэша проверены для `dinov2-small`, `dinov2-with-registers-small` и `timm/resnet18.a1_in1k`; DINOv3 проверен только на архитектуре (маленькая модель из transformers), реальные веса gated.

## Формат данных

Проект ожидает датасет в **YOLO-подобном** виде и YAML с метаданными (как в экосистеме Ultralytics).

### Файл `dataset.yaml`

- **`path`** — корень набора данных (папка с `train`/`valid`/…).
- **`train`** / **`val`** — подпути к изображениям **относительно** `path` (например `train/images`, `valid/images`).
- **`nc`** — число классов.
- **`names`** — имена классов: список или словарь `{индекс: имя}`.

`path` разрешается так же, как в Ultralytics: абсолютный путь, затем относительно текущей папки, затем относительно папки с `dataset.yaml`. Пути `train` / `val` берутся из YAML, каталог разметки получается заменой `images` на `labels`.

### Дерево папок (поддерживаются два варианта)

**Вариант 1:** split внутри корня

```
<root>/train/images/<name>.jpg
<root>/train/labels/<name>.txt
<root>/valid/images/...
<root>/valid/labels/...
```

Если ключа `val:` нет, используется `valid/images`.

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

- На вход модели изображение нормализуется **ImageNet** `mean` / `std`; размер задаётся `imgsz` (**letterbox** до квадрата, см. [`dataset.py`](centernet/dataset.py)).

## Обучение, валидация, инференс

После `pip install -e .` доступна команда `centernet` (или `python -m centernet`), аргументы в виде `key=value`, как у `yolo`:

```bash
centernet train   data=detection_dataset_yolo/dataset.yaml model=facebook/dinov2-small epochs=50 batch=16 imgsz=640
centernet val     model=runs/train/exp/best.pt data=detection_dataset_yolo/dataset.yaml
centernet predict model=runs/train/exp/best.pt source=path/to/image_or_dir conf=0.25
```

То же из Python:

```python
from centernet import CenterNet

model = CenterNet("facebook/dinov2-small")      # или CenterNet("runs/train/exp/best.pt")
model.train(data="detection_dataset_yolo/dataset.yaml", epochs=50)
metrics = model.val()                           # dict: loss, map50, map50_95, mar_100
dets = model.predict("img.jpg", conf=0.25)      # list[Tensor[N, 6]]: x1, y1, x2, y2, score, class
```

Все ключи и значения по умолчанию - в [`centernet/cfg/default.yaml`](centernet/cfg/default.yaml); свой файл подключается как `cfg=my.yaml` (поверх него действуют аргументы командной строки). Неизвестный ключ вызывает ошибку. `centernet help` выводит краткую справку.

### Дообучение и resume

**Дообучение** с уже обученной модели вместо чистого DINOv2:

```bash
centernet train model=runs/train/exp/best.pt data=new_dataset.yaml epochs=30
```

Бэкбон, neck и голова берутся из чекпоинта. Если в новом датасете другое число классов, переносится всё, что совпадает по форме, а заново создаётся только последний слой heatmap; печатается `Transferred N/M items`. Без `data=` обучение пойдёт на датасете из чекпоинта.

**Resume** прерванного обучения (как в Ultralytics):

```bash
centernet train resume=runs/train/exp/last.pt
```

Все аргументы берутся из чекпоинта, поменять можно только `device`, `workers`, `batch`. Обучение продолжается со следующей эпохи в той же папке, `results.csv` дописывается. `last.pt` хранит всё для продолжения: обычные и EMA-веса, optimizer, scheduler, GradScaler, лучшую метрику и состояние ранней остановки; `best.pt` - только EMA-веса для инференса. Resume уже завершённого обучения ничего не делает. Состояние генераторов случайных чисел не восстанавливается, поэтому порядок батчей и аугментации после resume будут другими, чем без прерывания.

### Аугментации

Как в Ultralytics, задаются аргументами `train` (значения по умолчанию те же, что у YOLO), применяются только к train-выборке:

| Ключ | По умолчанию | Что делает |
|------|--------------|------------|
| `mosaic` | 1.0 | вероятность mosaic из 4 картинок |
| `close_mosaic` | 10 | последние N эпох без mosaic / mixup / copy_paste |
| `mixup` | 0.0 | вероятность смешать с другой (тоже аугментированной) картинкой |
| `copy_paste` | 0.0 | вероятность вставить каждый объект другой картинки прямоугольной вырезкой (масок нет), если он закрывает существующие объекты < 30% |
| `degrees`, `translate`, `scale`, `shear`, `perspective` | 0, 0.1, 0.5, 0, 0 | поворот, сдвиг, масштаб, скос, перспектива одной матрицей |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.015, 0.7, 0.4 | оттенок, насыщенность, яркость |
| `flipud`, `fliplr` | 0.0, 0.5 | вероятности отражений |

```bash
centernet train data=dataset.yaml mosaic=1.0 mixup=0.1 copy_paste=0.3 degrees=10 flipud=0.5 hsv_v=0.3
```

Все трансформации - в [`centernet/augment.py`](centernet/augment.py), боксы пересчитываются вместе с пикселями (это проверяют тесты по цветным прямоугольникам). Отключить всё: `mosaic=0 translate=0 scale=0 hsv_h=0 hsv_s=0 hsv_v=0 fliplr=0`.

### Learning rate и AMP

- `lr0` - стартовый lr головы, `lr_backbone` - бэкбона (когда он разморожен), конечный lr = стартовый × `lrf` (по умолчанию 0.01).
- `cos_lr=true` - косинусный спад, `false` (по умолчанию, как в Ultralytics) - линейный.
- `warmup_epochs=3` - линейный разогрев от 0 до lr0 по шагам оптимизатора; lr меняется на каждом шаге.
- `amp=true` (по умолчанию) - смешанная точность float16 на CUDA через `torch.autocast` + `GradScaler`; на CPU игнорируется.

```bash
centernet train data=dataset.yaml lr0=2e-4 lrf=0.05 cos_lr=true warmup_epochs=1 amp=true
```

На время обучения аллокатор CUDA ограничен 90% VRAM: кэш PyTorch бывает в 1.5 раза больше реально занятой памяти, и без лимита драйвер Windows уходит в общую память (итерации в разы медленнее).

### Что выводится

Во время эпохи tqdm показывает loss по батчам. После каждой эпохи печатается строка

```
epoch=3/50 | train_loss=21.4 | val_loss=19.8 | P=0.512 | R=0.430 | map50=0.401 | map50-95=0.233 | mar100=0.350 | lr=9.8e-05/4.9e-06 | 94s *
```

`P` и `R` считаются при IoU 0.5 и score ≥ 0.25, `lr` - голова/бэкбон, `*` - новая лучшая эпоха. В конце обучения `best.pt` валидируется ещё раз и печатается таблица AP50 / AP50-95 по классам (то же делает `centernet val`).

### Что сохраняется

Каталог запуска `runs/train/exp` (при повторе `exp2`, `exp3`, ...; `exist_ok=True` перезаписывает):

- `best.pt` - лучший по **mAP@0.50-0.95**, `last.pt` - последний. В чекпоинте лежат **EMA-веса** (`model`), `optimizer`, `scheduler`, `epoch`, `metrics` и `args` (полный конфиг запуска, `nc`, `names`), поэтому `val` и `predict` не требуют указывать архитектуру;
- `args.yaml` и `results.csv` (loss, precision, recall, mAP, lr и время по эпохам).

`predict` сохраняет картинки с боксами в `runs/predict/exp` (`save=False` отключает) и возвращает боксы в координатах исходного изображения. `val` по умолчанию использует `conf=0.001`.

Бэкбон заморожен, обучается голова. EMA считается через `torch.optim.swa_utils.AveragedModel` (вместе с буферами BatchNorm), метрики на валидации считаются по EMA-модели. `fraction=0.1` берёт 10% данных для быстрой проверки. EMA с прогревом (`decay` растёт от 0 до `ema_decay`).

**Ранняя остановка:** `patience=N` останавливает обучение, если mAP@0.50-0.95 не росла N эпох подряд (по умолчанию 50, `patience=0` выключает). Равенство не считается ростом.

**Авто-batch:** `batch=-1` на GPU подбирает наибольший batch по реальным пробным прогонам forward+backward (с проверкой OOM и лимитом 90% VRAM), затем берёт 85% от него. На CPU используется `nbs`. Batch всегда ограничен размером train-выборки.

Не реализовано: экспорт (ONNX), видео и веб-камера в `source`.

## Структура репозитория (основное)

| Файл | Роль |
|------|------|
| `centernet/cli.py` | разбор `centernet MODE key=value` |
| `centernet/engine.py` | класс `CenterNet`: `train`, `val`, `predict`, цикл обучения и метрики MAP |
| `centernet/cfg/default.yaml` | значения по умолчанию |
| `centernet/model.py` | DINOv2 + CenterNetHead, loss, decode пиков |
| `centernet/dataset.py` | `YOLODataset`, letterbox, collate |
| `centernet/loss.py` | фокальный лосс для heatmap |
| `centernet/utils.py` | гауссианы для целевых карт |
| `centernet/augment.py` | аугментации: mosaic, mixup, copy_paste, геометрия, HSV, отражения |
| `centernet/backbone.py` | адаптеры бэкбонов: ViT-семейство и иерархические + FPN |
| `tests/` | тесты: CLI/конфиг, бэкбоны, decode, выравнивание признаков, loss, аугментации, lr, AMP, early stopping, авто-batch, метрики |

## Лицензия

См. файл [`LICENSE`](LICENSE).
