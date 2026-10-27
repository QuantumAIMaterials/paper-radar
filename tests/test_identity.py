import pytest

from paper_radar.identity import normalize_doi, normalize_title


@pytest.mark.parametrize("value", [
    "10.1234/Graphene", " DOI:10.1234/Graphene ",
    "https://doi.org/10.1234/Graphene", "http://dx.doi.org/10.1234/Graphene",
])
def test_doi_formats_share_one_identity(value):
    assert normalize_doi(value) == "10.1234/graphene"


@pytest.mark.parametrize("value", [None, "None", "", 12, "not-a-doi"])
def test_invalid_doi_cannot_become_an_identity(value):
    assert normalize_doi(value) == ""


def test_title_normalization_keeps_scientific_letters():
    assert normalize_title("Transport in α phase") != normalize_title("Transport in β phase")
    assert (
        normalize_title("Electric-field transport") == normalize_title("Electric field transport")
    )
