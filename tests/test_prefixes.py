"""Unit checks for prefix and truncation helpers. No model download needed."""

from src.embed.prefixes import (
    format_query,
    format_doc,
    check_same_dim,
    truncate_and_renorm,
    pick_dtype,
    SUPPORTED_MRL_DIMS,
)


def test_search_prefix_byte_exact():
    out = format_query("SearchQuery", "GST refund kaise claim karein?")
    assert out == "task: search result | query: GST refund kaise claim karein?"


def test_doc_with_title():
    out = format_doc("GST refund", "apply in portal")
    assert out == "title: GST refund | text: apply in portal"


def test_doc_none_title():
    assert format_doc(None, "hello") == "title: none | text: hello"
    assert format_doc("  ", "hello") == "title: none | text: hello"


def test_same_dim_gate():
    check_same_dim(128, 128)
    try:
        check_same_dim(128, 768)
    except ValueError:
        return
    raise AssertionError("same dim gate did not fire")


def test_truncate_renorm_unit_length():
    import numpy as np

    v = np.ones(768)
    for d in SUPPORTED_MRL_DIMS:
        out = truncate_and_renorm(v, d)
        assert len(out) == d
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-9


def test_bad_dim_rejected():
    import numpy as np

    try:
        truncate_and_renorm(np.ones(768), 64)
    except ValueError:
        return
    raise AssertionError("bad dim passed")


def test_dtype_never_fp16():
    assert pick_dtype() in ("bfloat16", "float32")
    assert pick_dtype() != "float16"
