# tests/test_load_data.py
import pytest
from pathlib import Path
import pandas as pd
from src.data_processing import load_data  # Adjust path if needed


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    content = """TransactionId,TransactionStartTime,Amount
1,2023-01-01T12:00:00Z,1000.50
2,2023-01-02T14:30:00Z,-500.00
"""
    file_path = tmp_path / "sample.csv"
    file_path.write_text(content.strip())
    return file_path


def test_load_data_success(sample_csv: Path):
    df = load_data(sample_csv)
    assert len(df) == 2
    assert pd.api.types.is_datetime64tz_dtype(df["TransactionStartTime"])
    assert df["TransactionStartTime"].dt.tz is not None


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("fake_path.csv")