from src.train import load_data


def test_load_data_shape_and_target():
    df = load_data()
    assert df.shape == (2000, 9)
    assert set(df["target"].unique()) == {0, 1}
    assert df.isna().sum().sum() == 0
