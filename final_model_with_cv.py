from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
from data_augmentation import add_awgn, bandpass_filter, time_shift, pitch_shift
from feature_extraction import extract_features
from CNN import MultiFeatureCNN

# Функция запускает кросс-валидацию по 5 фолдам (или LeaveOneOut). Для каждого фолда трейн аугментируется, потом считаются признаки,
# создается CNN, производится классификация с помощью CNN и XGBoost, с помощью мягкого голосования получаются итоговые предсказания.


def run_cv_with_aug(df, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results_xgd, results_cnn, results_ens = [], [], []
    # skf = LeaveOneOut()
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['signal'], df['state'])):
        print(f"\n Fold {fold+1}")

        # Аугментируем трейн и сразу считаем признаки.

        X_cnn_mfcc, X_cnn_mel, X_cnn_chroma, X_cnn_rol, X_cnn_zcr, X_xgb, y = [], [], [], [], [], [], []

        for idx in train_idx:
            signal = df['signal'].iloc[idx]
            label = df['state'].iloc[idx]

            feats = extract_features(signal)
            X_cnn_mfcc.append(feats['mfcc'])
            X_cnn_mel.append(feats['mel'])
            X_cnn_chroma.append(feats['chroma'])
            X_cnn_rol.append(feats['rolloff'])
            X_cnn_zcr.append(feats['zcr'])
            X_xgb.append(feats['x_xgb'])
            y.append(label)

            for aug_sig in [add_awgn(signal), bandpass_filter(signal), time_shift(signal), pitch_shift(signal)]:
                feats_aug = extract_features(aug_sig)
                X_cnn_mfcc.append(feats_aug['mfcc'])
                X_cnn_mel.append(feats_aug['mel'])
                X_cnn_chroma.append(feats_aug['chroma'])
                X_cnn_rol.append(feats_aug['rolloff'])
                X_cnn_zcr.append(feats_aug['zcr'])
                X_xgb.append(feats_aug['x_xgb'])
                y.append(label)
            for aug_sig in [add_awgn(signal), bandpass_filter(signal), time_shift(signal), pitch_shift(signal)]:
                feats_aug = extract_features(aug_sig)
                X_cnn_mfcc.append(feats_aug['mfcc'])
                X_cnn_mel.append(feats_aug['mel'])
                X_cnn_chroma.append(feats_aug['chroma'])
                X_cnn_rol.append(feats_aug['rolloff'])
                X_cnn_zcr.append(feats_aug['zcr'])
                X_xgb.append(feats_aug['x_xgb'])
                y.append(label)

        # Считаем признаки для валидационной выборки

        X_val_mfcc, X_val_mel, X_val_chroma, X_val_rol, X_val_zcr, X_val_xgb, y_val = [], [], [], [], [], [], []

        for idx in val_idx:
            signal = df['signal'].iloc[idx]
            label = df['state'].iloc[idx]
            feats = extract_features(signal)

            X_val_mfcc.append(feats['mfcc'])
            X_val_mel.append(feats['mel'])
            X_val_chroma.append(feats['chroma'])
            X_val_rol.append(feats['rolloff'])
            X_val_zcr.append(feats['zcr'])
            X_val_xgb.append(feats['x_xgb'])
            y_val.append(label)

        # Преобразуем получившиеся признаки в нужный формат

        X_cnn_mfcc = np.array(X_cnn_mfcc)[..., np.newaxis]
        X_cnn_mel = np.array(X_cnn_mel)[..., np.newaxis]
        X_cnn_chroma = np.array(X_cnn_chroma)[..., np.newaxis]
        X_cnn_rol = np.array(X_cnn_rol)[..., np.newaxis]
        X_cnn_zcr = np.array(X_cnn_zcr)[..., np.newaxis]
        X_xgb = np.array(X_xgb)
        y = np.array(y)

        X_val_mfcc = np.array(X_val_mfcc)[..., np.newaxis]
        X_val_mel = np.array(X_val_mel)[..., np.newaxis]
        X_val_chroma = np.array(X_val_chroma)[..., np.newaxis]
        X_val_rol = np.array(X_val_rol)[..., np.newaxis]
        X_val_zcr = np.array(X_val_zcr)[..., np.newaxis]
        X_val_xgb = np.array(X_val_xgb)
        y_val = np.array(y_val)

        X_cnn_rol = np.expand_dims(X_cnn_rol, axis=-1)
        X_cnn_zcr = np.expand_dims(X_cnn_zcr, axis=-1)
        X_val_rol = np.expand_dims(X_val_rol, axis=-1)
        X_val_zcr = np.expand_dims(X_val_zcr, axis=-1)

        # Пять сверточных сетей на каждую группу признаков и их комбинация в итоговую сеть.

        input_shapes = [
            X_cnn_mfcc.shape[1:], X_cnn_mel.shape[1:], X_cnn_chroma.shape[1:], X_cnn_rol.shape[1:], X_cnn_zcr.shape[1:]

        ]

        cnn_inputs = [X_cnn_mfcc, X_cnn_mel, X_cnn_chroma, X_cnn_rol, X_cnn_zcr]
        cnn_val_inputs = [X_val_mfcc, X_val_mel, X_val_chroma, X_val_rol, X_val_zcr]

        cnn_ensemble = MultiFeatureCNN(input_shapes)

        history = cnn_ensemble.fit(
            cnn_inputs, y,
            validation_data=(cnn_val_inputs, y_val),
            epochs=200, batch_size=64, verbose=1
        )

        y_pred_cnn = cnn_ensemble.predict(cnn_val_inputs).squeeze()

        # Классификация с помощью XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=400, eval_metric='logloss', learning_rate=0.3,)
        xgb_model.fit(X_xgb, y)
        y_pred_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]

        # Мягкое голосование и подсчёт точности
        alpha = 0.5
        xgb_final = (y_pred_xgb) > 0.5
        cnn_final = (y_pred_cnn) > 0.5
        ens_y_final = (alpha * y_pred_cnn + (1 - alpha) * y_pred_xgb) > 0.5
        # y_final = (y_pred_xgb)>0.5
        xgb_acc = accuracy_score(y_val, xgb_final)
        cnn_acc = accuracy_score(y_val, cnn_final)
        ens_acc = accuracy_score(y_val, ens_y_final)

        print(f"Fold accuracy XGBmodel: {xgb_acc:.4f}")
        print(f"Fold accuracy CNNmodel: {cnn_acc:.4f}")
        print(f"Fold accuracy Ensemble: {ens_acc:.4f}")
        results_xgd.append(xgb_acc)
        results_cnn.append(cnn_acc)
        results_ens.append(ens_acc)

    print(f"\nMean CV accuracy XGBmodel: {np.mean(results_xgd):.4f}")
    print(f"\nMean CV accuracy CNNmodel: {np.mean(results_cnn):.4f}")
    print(f"\nMean CV accuracy Ensemble: {np.mean(results_ens):.4f}")

# Предварительно обрежем начало пневмограмм так, чтобы все сигналы имели одинаковую длину.
df = pd.read_parquet('/content/initial_data.parquet') # Исходные данные
for i in range(80):
  df['signal'][i] = df['signal'][i][-11933:]

# Применение функции
run_cv_with_aug(df)