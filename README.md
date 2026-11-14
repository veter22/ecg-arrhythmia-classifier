# ECG Arrhythmia Classifier (MIT-BIH)

Сквозной (end-to-end) ML-пайплайн для автоматической детекции R-пиков и классификации сердечных аритмий по одноканальной ЭКГ в соответствии со стандартом **AAMI EC57** (`N`, `S`, `V`, `F`).

## Ключевые инженерные решения
1. **Пациентонезависимый сплит (Inter-patient split):** Строгое разделение по ID пациентов (`DS1` — обучение, `DS2` — тест по схеме *de Chazal et al.*), исключающее утечку морфологии ЭКГ одного пациента между выборками.
2. **Автономная детекция пиков:** Использование полосового фильтра Баттерворта (0.5–40 Гц) и алгоритма **Пана — Томпкинса** вместо опоры на готовые аннотации кардиологов в инференсе.
3. **Гибридная архитектура (`1D CNN + RR-MLP`):** Объединение морфологических признаков формы QRS-комплекса (окно 256 отсчётов) с контекстом ритма (`pre_RR`, `post_RR`, `local_RR`, `ratio_RR`).
4. **Борьба с дисбалансом без убийства Precision:** Использование **Focal Loss** ($\gamma = 2.0$) со сглаженными весами по эффективному числу сэмплов ($eta = 0.999$) и оценка по **PR-AUC** и **Macro-F1**.

---

## Команда и распределение обязанностей

| Роль | Участник | Ветка в Git | Зона ответственности | Файлы |
| :--- | :--- | :--- | :--- | :--- |
| **Data & Signal Processing Lead** | `@member1` *(вписать имя/ник)* | `feature/data-pipeline` | Загрузка MIT-BIH, маппинг AAMI, фильтрация, алгоритм Пана — Томпкинса, расчёт RR-фичей, сборка `Dataset` | `configs/config.yaml`, `src/data/*` |
| **ML & Evaluation Lead** | `@member2` *(вписать имя/ник)* | `feature/ml-model` | Архитектура `HybridECGClassifier`, `FocalLoss`, цикл обучения, валидация, расчёт PR-AUC и Confusion Matrix, error analysis | `src/models/*`, `src/losses/*`, `src/train.py`, `src/evaluate.py`, `notebooks/*` |

> **Точка стыка (Контракт данных):** описана в `src/contracts.py`. Оба участника обязаны соблюдать формат батча `{"signal": [B, 1, 256], "rr": [B, 4], "label": [B]}`.

---

## Быстрый старт и настройка Git

```bash
# 1. Инициализировать репозиторий и создать рабочие ветки
bash init_git.sh

# 2. Создать виртуальное окружение и поставить зависимости
python -m venv .venv
source .venv/bin/activate  # для Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Проверить работу скелета нейросети (smoke-test)
python -m src.models.hybrid_cnn
```

### Как залить на GitHub / GitLab
```bash
git remote add origin <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
git push -u origin main
git push -u origin feature/data-pipeline
git push -u origin feature/ml-model
```

---

## Целевые метрики приёмки (на тесте DS2)
- **Macro F1:** $\ge 0.72$
- **Класс `V` (желудочковые экстрасистолы):** Recall $\ge 0.85$, Precision $\ge 0.80$
- **Класс `S` (наджелудочковые экстрасистолы):** Recall $\ge 0.65$, Precision $\ge 0.60$

## Лицензия и цитирование данных
Датасет MIT-BIH Arrhythmia Database распространяется по лицензии **ODC-By 1.0** (PhysioNet).
*Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).*
