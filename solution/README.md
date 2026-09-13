# LPR-RU: распознавание нестандартных ГРЗ (типы 1, 1А, 1Б) — полуфинал Volga IT 2026

Решение принимает каталог с изображениями и формирует CSV `image;plate_num;plate_type;confidence`
(раздел 5 задания). Работает полностью офлайн, все веса лежат в `weights/`.

## Быстрый старт

```bash
# 1. окружение (Python 3.9–3.12)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU (GTX 1050 Ti); для CPU: pip install torch torchvision
pip install -r requirements.txt

# 2. запуск на каталоге изображений  (ТОЧНАЯ КОМАНДА ЗАПУСКА)
python run.py --input /path/to/images --output result.csv
```

Путь до каталога можно передать тремя способами: аргумент `--input`, переменная окружения
`LPR_INPUT` или ключ `input:` в `config.yaml`. Выход — `--output` / `LPR_OUTPUT` (по умолчанию `result.csv`).
Устройство выбирается автоматически (`cuda:0` → `mps` → `cpu`), можно задать `--device cpu`.

Пример вывода:

```
image;plate_num;plate_type;confidence
img_0042.jpg;A123BC716;type1a;0.9312
img_0043.jpg;K555OP25;type1b;0.8801
img_0044.jpg;H741C###;type1;0.6204
```

## Состав репозитория

| Путь | Назначение |
|---|---|
| `run.py` | CLI: каталог изображений → CSV |
| `config.yaml` | параметры по умолчанию (пороги, размер входа, гейт «знак на ТС») |
| `lpr/` | библиотека: `detector.py` (YOLO11-pose, 4 угла знака + класс типа), `ocr.py` (CRNN+CTC), `pipeline.py`, `plates.py` (маски ГОСТ) |
| `weights/` | `det.pt` (детектор), `ocr.pt` (распознаватель), `yolo11n.pt` (COCO, гейт «знак на ТС»), `face_detection_yunet_2023mar.onnx` (размытие лиц при сборке датасета) |
| `generator/` | генератор синтетики (`generate.py`, `plate_render.py`, шрифты + лицензии) — копия лежит в `dataset/generator` |
| `train/` | `train_detector.py`, `train_ocr.py` — скрипты обучения |
| `tools/` | сбор данных (`harvest_commons.py`), полуавтоматическая разметка (`autolabel.py`, `review_sheets.py`), сборка датасета (`build_dataset.py`), `validate_dataset.py`, `evaluate.py`, `benchmark.py`, `blur_faces.py` |
| `docs/` | пояснительная записка |

## Как это работает

1. **Детектор** — YOLO11n-pose, дообученный на 4 класса (`type1`, `type1a`, `type1b`, `other`) с 4 ключевыми
   точками = углами знака. Один проход даёт bbox, тип и точный четырёхугольник для выравнивания перспективы.
2. **Выравнивание** — гомография по 4 углам в канонический прямоугольник (520×112 или 290×170 в пикселях
   208×48 / 192×112). Для квадратного знака 1А прямоугольник режется на две строки.
3. **Распознавание** — компактный CRNN (CNN → BiLSTM → CTC), вход 32×160, алфавит `0-9`, `ABEKMHOPCTYX`, `#`.
   Символы с уверенностью ниже порога заменяются на `#`; символ `#` также является отдельным классом,
   которому модель обучена на закрытых/грязных позициях.
4. **Грамматика** — маска `Б ЦЦЦ ББ РР(Р)` (типы 1, 1А, 1Б по заданию) и маска `ББ ЦЦЦ РР(Р)` (тип 1Б по
   ГОСТ Р 50577-2018). Позиционное исправление похожих символов (O↔0, B↔8 …); неправдоподобная строка
   снижает confidence / переводит знак в `other`.
5. **Гейт «знак на ТС»** — COCO-детектор YOLO11n (car/bus/truck/motorcycle): знак вне ТС (щит, витрина,
   экран) получает пониженную уверенность (или не выводится с `--drop-off-vehicle`).

## Обучение (воспроизведение)

```bash
# 1. синтетические сцены (детерминированы seed'ом) — уже лежат в dataset/images/synthetic
python generator/generate.py --out ../dataset --n 5200 --seed 42 --backgrounds ../dataset/images/real --meta ../dataset/meta.csv --procedural-frac 0.25
# 2. кропы для распознавателя: синтетические + реальные из датасета
python generator/generate.py --mode crops --out work/ocr_syn --ocr-crops work/ocr_syn --n 60000 --seed 11
python tools/export_ocr_crops.py --dataset ../dataset --out work/ocr_real --prefix real --jitter 3
# 3. детектор (labels/*.txt в формате YOLO-pose; 15 % реальных изображений откладываются для теста)
python train/train_detector.py --data ../dataset --epochs 40 --imgsz 640 --holdout-frac 0.15 --out weights/det.pt
# 4. распознаватель
python train/train_ocr.py --crops work/ocr_syn --crops "work/ocr_real:4" --epochs 30 --out weights/ocr.pt
```

Полный пайплайн сборки датасета из проверенных кандидатов и генерации синтетики — `tools/build_all.sh`;
итеративная разметка — `tools/autolabel.py` → `tools/review_sheets.py` → `tools/apply_notes.py` → `tools/build_dataset.py`.

## Проверка скорости

```bash
python tools/benchmark.py --input /path/to/images --device cuda:0
```

Размер входа детектора задаётся `imgsz` в `config.yaml` (по умолчанию 960: находит знаки шириной от ~30 px).
Замеренная скорость на Apple M4: 78 мс/изображение (медиана, GPU) и 92 мс на одном только CPU — требование
100 мс выполняется в обоих случаях. Ускорить дальше: `--imgsz 640` (≈2× быстрее), `--no-vehicle-gate`
(−20 мс), `--half` (FP16 на CUDA). Подробные замеры — в пояснительной записке (`docs/note.md`).

## Оценка качества

```bash
# отложенная реальная выборка (15 % изображений датасета) и метрики
bash tools/eval_holdout.sh ../dataset cuda:0
# сравнение произвольного result.csv с эталоном в формате задания
python tools/evaluate.py --gt gt.csv --pred result.csv
```

## Лицензии

Код решения — MIT. Веса `yolo11n*.pt` и библиотека Ultralytics — AGPL-3.0 (некоммерческое использование).
Шрифт `GOST-R-50577-93.ttf` — «Do What You Want Public License» (stanlapru); Roboto Condensed, Oswald,
PT Sans Narrow — SIL OFL 1.1. YuNet — MIT. Датасет — CC BY 4.0 (см. `dataset/LICENSE`, `dataset/attribution.csv`).

## Самопроверка установки

```bash
python run.py --input samples --output samples_result.csv && cat samples_result.csv
```

В `samples/` лежат четыре фотографии из датасета (по одной на тип; лицензии в `samples/ATTRIBUTION.csv`);
ожидаемые номера указаны там же.
