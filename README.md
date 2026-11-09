# SyntheDator

**Synthetic-data experiments for causal analysis and variance reduction.** This repository preserves two attributed collections of generators and exploratory notebooks. It is a research collection, without a packaged API or a production data-generation service.

SyntheDator объединяет ранние эксперименты с синтетическими данными: признаки с
лагами, потенциальные исходы, механизм получения воздействия и CUPAC-корректировку.
Здесь важно видеть конкретную модель генерации и ограничения эксперимента.

## С чего начать

| Материал | Что в нём |
|---|---|
| [`by_ivan/data_gen.py`](https://github.com/Yurashku/SyntheDator/blob/main/old_collections/by_ivan/data_gen.py) | `DataGenerator`: нормальные, бинарные и gamma-признаки, лаги, скрытая переменная, назначение воздействия и потенциальные исходы. |
| [`by_ivan/autocupac.py`](https://github.com/Yurashku/SyntheDator/blob/main/old_collections/by_ivan/autocupac.py) | `CUPACTransformer`: сравнение моделей на фолдах, корректировка целевой переменной и текстовый отчёт. |
| [`by_ivan/runner.ipynb`](https://github.com/Yurashku/SyntheDator/blob/main/old_collections/by_ivan/runner.ipynb) и [`Synthetic_datamaker.ipynb`](https://github.com/Yurashku/SyntheDator/blob/main/old_collections/by_ivan/Synthetic_datamaker.ipynb) | Исследовательские ноутбуки. |
| [`by_dmatryus`](https://github.com/Yurashku/SyntheDator/tree/main/old_collections/by_dmatryus) | Отдельная коллекция распределений и генератор таблиц `Syntdator`; происхождение сохранено в структуре репозитория. |

## Первый запуск без внешних данных

Для генератора из `by_ivan/data_gen.py` нужны только NumPy, pandas и SciPy:

```bash
git clone https://github.com/Yurashku/SyntheDator.git
cd SyntheDator
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy
```

Сохраните следующий пример в `example.py` в корне репозитория и выполните
`python example.py`:

```python
import numpy as np
from old_collections.by_ivan.data_gen import DataGenerator

data = DataGenerator(n_samples=1000, seed=7).generate()
print("shape:", data.shape)
print("columns:", ", ".join(data.columns))
print("observed outcome matches:", bool(np.allclose(
    data["y"], np.where(data["d"].eq(1), data["y1"], data["y0"])
)))
print("finite:", bool(np.isfinite(data.to_numpy()).all()))
again = DataGenerator(n_samples=1000, seed=7).generate()
print("seed repeatable:", data.equals(again))
```

Проверенный вывод на Python 3.14.0, NumPy 2.5.4, pandas 3.1.1 и SciPy 1.18.1:

```text
shape: (1000, 13)
columns: X1, X1_lag, X2, X2_lag, y0, y0_lag_1, y0_lag_2, z, U, D, d, y1, y
observed outcome matches: True
finite: True
seed repeatable: True
```

`y0` и `y1` — синтетические потенциальные исходы; `y` выбирается по фактическому
воздействию `d`. `z`, `U` и `D` участвуют в механизме назначения, а `d = D * z`.
Параметр `effect_size` — слагаемое в формуле `y1`; индивидуальная разность
`y1 - y0` также содержит вклад `U` и шум, поэтому она не постоянна.

## Границы эксперимента

- Это проверка схемы, конечности чисел, связи наблюдаемого исхода с потенциальными
  и повторяемости при seed. Она не доказывает калибровку методов причинного анализа.
- Скрытая переменная `U` включена в синтетическую таблицу для проверки механизма.
  Не подавайте её или недоступные потенциальные исходы в модель, если эксперимент
  должен воспроизводить их ненаблюдаемость.
- Для gamma-переменных корреляция задаётся через латентную нормальную пару; параметр
  не следует автоматически считать Pearson-корреляцией итоговых gamma-значений.
- Генератор использует глобальный NumPy seed. Ноутбуки и CUPAC-пайплайн имеют
  дополнительные зависимости (scikit-learn, CatBoost, Jupyter) и в этой проверке
  не исполнялись. Их результаты требуют отдельной проверки выбранного дизайна.

## Происхождение и лицензия

Коллекции `by_ivan` и `by_dmatryus` сохранены раздельно. Вклад Yurashku можно
проследить по [изменению генератора](https://github.com/Yurashku/SyntheDator/commit/d6d5d481c2c7a4500dc15311ba72f9c1718db743),
[изменению CUPAC](https://github.com/Yurashku/SyntheDator/commit/c7f19118d1e5c33be1d81d416ca720136600527d)
и [общей истории](https://github.com/Yurashku/SyntheDator/commits/main/).
Эти ссылки не приписывают Ивану авторство всей второй коллекции.

Лицензия репозитория — [Apache 2.0](https://github.com/Yurashku/SyntheDator/blob/main/LICENSE).
