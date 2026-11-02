import pytest

from ml_failure_lab.cases.memorization import experiment, require_disjoint_evaluation


def test_memorization_bad_approach_overlaps_fit_rows():
    res = experiment("bad")
    assert res.fit_rows == res.evaluated_rows
    assert res.accuracy == 1.0
    with pytest.raises(ValueError, match="overlapping records"):
        require_disjoint_evaluation(res)


def test_memorization_fixed_approach_is_disjoint():
    res = experiment("fixed")
    assert res.fit_rows.isdisjoint(res.evaluated_rows)
    assert 0.0 <= res.accuracy <= 1.0
    require_disjoint_evaluation(res)


def test_memorization_unknown_approach():
    with pytest.raises(ValueError, match="Unknown approach"):
        experiment("invalid")
