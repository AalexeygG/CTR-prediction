# Ad CTR Prediction with SVM

Предсказание вероятности клика по рекламному баннеру. Линейный SVC с изотонической калибровкой вероятностей.

Данные: 50 000 показов рекламы, доля кликов около 17%. Основная метрика PR-AUC.

## Что сделано

- EDA, удаление `id` и `device_ip`, час суток из `hour`, флаг для значения -1 в `C20`
- разбиение 80/20 со стратификацией
- Pipeline: StandardScaler, OneHotEncoder, TargetEncoder, VarianceThreshold, SelectKBest (k=30)
- сравнение DummyClassifier, LogisticRegression и SVC на кросс-валидации, GridSearchCV
- калибровка SVC через CalibratedClassifierCV (isotonic, cv=5), метрики Brier, ECE, MCE

## Результаты на тесте

| Модель | PR-AUC | Brier |
|---|---|---|
| DummyClassifier | 0.1721 | 0.1721 |
| LogisticRegression | 0.3971 | 0.1252 |
| SVC до калибровки | 0.2676 | 0.1519 |
| SVC после калибровки | 0.3506 | 0.1307 |

Калибровка SVC: ECE 0.0968 → 0.0232, Brier 0.1519 → 0.1307.

## Запуск

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Python 3.9+.

## Структура

```
ctr_prediction.ipynb   код, графики и выводы
requirements.txt       версии библиотек
datasets/              ds_s16_ad_click_dataset.csv
models/                сохранённые модели
```
