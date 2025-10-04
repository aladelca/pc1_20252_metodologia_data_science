import pandas as pd
import os
from src.preprocess.cleaning import clean_for_day_and_category_aggregation
from src.preprocess.aggregation import aggregate_by_day_and_category
from src.preprocess.preprocessing import fill_missing_dates_with_zero, preprocessing

def test_clean_for_day_and_category_aggregation():
    data = {
        "transaction_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "product_category": ["A", "(not set)", None],
        "product_quantity": [10, 5, None],
    }
    df = pd.DataFrame(data)

    result = clean_for_day_and_category_aggregation(df)

    assert "transaction_date" in result.columns
    assert result["product_category"].dtype.name == "category"
    assert len(result) == 1
    assert result["product_category"].iloc[0] == "A"

def test_aggregate_by_day_and_category():
    data = {
        "transaction_date": pd.to_datetime(
            ["2023-01-01", "2023-01-01", "2023-01-02"]
        ),
        "product_category": ["A", "A", "A"],
        "product_quantity": [5, 7, 3],
    }
    df = pd.DataFrame(data)

    result = aggregate_by_day_and_category(df)

    assert "product_quantity" in result.columns
    assert result.loc[result["transaction_date"] == "2023-01-01", "product_quantity"].iloc[0] == 12
    assert result.loc[result["transaction_date"] == "2023-01-02", "product_quantity"].iloc[0] == 3
    
def test_fill_missing_dates_with_zero():
    data = {
        "transaction_date": pd.to_datetime(["2023-01-01", "2023-01-03"]),
        "product_category": ["A", "A"],
        "product_quantity": [5, 10],
    }
    df = pd.DataFrame(data)

    result = fill_missing_dates_with_zero(df)

    assert len(result) == 3
    assert result.loc[result["transaction_date"] == "2023-01-02", "product_quantity"].iloc[0] == 0.0


def test_preprocessing_integration(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    aggregated_dir = tmp_path / "data" / "aggregated"
    aggregated_dir.mkdir(parents=True)

    df = pd.DataFrame({
        "transaction_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "product_category": ["A", "A"],
        "product_quantity": [10, 15],
    })
    parquet_path = raw_dir / "data_sample.parquet"
    df.to_parquet(parquet_path)

    monkeypatch.chdir(tmp_path)
    result = preprocessing()

    assert result == "success"
    output_path = aggregated_dir / "day_category.parquet"
    assert output_path.exists()

    df_out = pd.read_parquet(output_path)
    assert not df_out.empty
    assert "product_quantity" in df_out.columns