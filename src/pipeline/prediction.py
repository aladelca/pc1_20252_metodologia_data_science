import joblib
from src.pipeline.preprocess import load_and_preprocess, split_train_test


def main():
    # Cargar datos
    df_no_quant, df_quant = (
        load_and_preprocess(
            "data/aggregated/day_category.parquet", "Accessories"
            )
        )

    # Split igual que en entrenamiento (80/20)
    _, X_test, _, y_test = split_train_test(
        df_no_quant, df_quant, train_size=0.8
        )

    # Cargar modelo
    gb = joblib.load("src/models/gb_model.pkl")

    # Predecir
    preds = gb.predict(X_test)

    print("✅ Predicciones generadas para el 1% final del dataset")
    for date, real, pred in zip(X_test.index[:49], y_test[:49], preds[:49]):
        print(f"{date.date()} | Real: {real} | Pred: {pred: .2f}")


if __name__ == "__main__":
    main()
