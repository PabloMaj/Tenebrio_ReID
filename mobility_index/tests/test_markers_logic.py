import pytest

from mobility.markers import LABEL_TO_ID, _order_tag, _sector


@pytest.mark.parametrize(
    "head, abdomen, expected",
    [
        ([0, 0], [10, 0], "I"),     # head row < abdomen row, vertical-ish
        ([10, 0], [0, 0], "II"),    # head row > abdomen row
        ([0, 0], [0, 10], "III"),   # head col < abdomen col, horizontal-ish
        ([0, 10], [0, 0], "IV"),    # head col > abdomen col
    ],
)
def test_sector(head, abdomen, expected):
    assert _sector(head, abdomen) == expected


def test_order_tag_sector_I_uses_column_order():
    r, w = LABEL_TO_ID["red"], LABEL_TO_ID["white"]
    # cluster1 left of cluster2 (smaller col) -> initial1 first
    assert _order_tag("I", r, w, (0, 0), (0, 10)) == "r_w"
    assert _order_tag("I", r, w, (0, 10), (0, 0)) == "w_r"


def test_order_tag_sector_II_is_reversed_vs_I():
    r, w = LABEL_TO_ID["red"], LABEL_TO_ID["white"]
    assert _order_tag("II", r, w, (0, 0), (0, 10)) == "w_r"


def test_order_tag_sectors_III_IV_use_row_order():
    g, b = LABEL_TO_ID["gold"], LABEL_TO_ID["blue"]
    assert _order_tag("III", g, b, (0, 0), (10, 0)) == "b_g"
    assert _order_tag("IV", g, b, (0, 0), (10, 0)) == "g_b"
