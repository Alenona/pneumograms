# pneumograms
Загрузка данных: download_initial_data.py
Parquet-файл со всеми исходными данными: initial_data.parquet

Вычисление характеристик:
Преобразование Фурье: FFT Characteristics.py -> FFT_Characteristics.parquet
Вейвлет-преобразование: CWT_Characteristics.py -> CWT_Characteristics.parquet
MFCC: MFCC.py -> MFCC.parquet
Аппроксимация спектра: spec approximation.py -> spec_approximation.parquet

Таблица со всеми характеристиками: total_features.py -> total_features.parquet

Обучение моделей: models_state.py, models_gender.py

Итоговые точности: accuracies_state.csv, accuracies_gender.csv
