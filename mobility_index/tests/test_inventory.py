import pandas as pd

from mobility.inventory import InventoryWriter, read_inventory


def test_write_read_roundtrip(tmp_path):
    inv = InventoryWriter(tmp_path / "inv.csv", columns=["series", "frame_idx", "tag", "x", "y"])
    inv.add(series="seria_2", frame_idx=1, tag="g_w", x=100, y=200)
    inv.add(series="seria_2", frame_idx=2, tag="r_b", x=110, y=205)
    assert len(inv) == 2
    path = inv.save()
    df = read_inventory(path)
    assert list(df.columns) == ["series", "frame_idx", "tag", "x", "y"]
    assert df.iloc[1]["tag"] == "r_b"
    assert df.iloc[0]["x"] == 100


def test_missing_columns_are_filled(tmp_path):
    inv = InventoryWriter(tmp_path / "inv.csv", columns=["a", "b", "c"])
    inv.add(a=1, b=2)  # c missing
    df = inv.to_frame()
    assert list(df.columns) == ["a", "b", "c"]
    assert pd.isna(df.iloc[0]["c"])
