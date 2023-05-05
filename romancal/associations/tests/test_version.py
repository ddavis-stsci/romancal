"""Test versioning consistency"""

from romancal import __version__
from romancal.associations.asn_from_list import asn_from_list


def test_asn_version():
    """Test version in association is package version"""

    asn = asn_from_list(["a", "b", "c"], product_name="aproduct")

    assert asn["code_version"] == __version__
