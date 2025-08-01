"""Roman tests for generating associations based on skycells using
a two step process. The first creates match files with a list of the patch
indexes. The second step reads these match files and generates a
set of association files based the patch index.
"""

import os

import pytest

from romancal.associations import mk_skycell_asn_from_skycell_list, mk_skycell_list

# mark all tests in this module
pytestmark = [pytest.mark.bigdata]

EXPECTED_MATCH_FILENAMES = [
    "r0000101001001001001_0002_wfi01_f158_cal.match",
    "r0000101001001001001_0002_wfi10_f158_cal.match",
]

EXPECTED_ASN_FILENAMES = [
    "r00001_p_v01001001001001_270p65x69y48_f158_asn.json",
    "r00001_p_v01001001001001_270p65x70y48_f158_asn.json",
    "r00001_p_v01001001001001_270p65x71y48_f158_asn.json",
    "r00001_p_v01001001001001_270p65x69y49_f158_asn.json",
    "r00001_p_v01001001001001_270p65x70y49_f158_asn.json",
    "r00001_p_v01001001001001_270p65x71y49_f158_asn.json",
    "r00001_p_v01001001001001_270p65x69y50_f158_asn.json",
    "r00001_p_v01001001001001_270p65x70y50_f158_asn.json",
    "r00001_p_v01001001001001_270p65x71y50_f158_asn.json",
    "r00001_p_v01001001001001_270p65x69y51_f158_asn.json",
    "r00001_p_v01001001001001_270p65x70y51_f158_asn.json",
    "r00001_p_v01001001001001_270p65x71y51_f158_asn.json",
    "r00001_p_v01001001001001_270p65x69y52_f158_asn.json",
    "r00001_p_v01001001001001_270p65x70y52_f158_asn.json",
    "r00001_p_v01001001001001_270p65x71y52_f158_asn.json",
]


@pytest.fixture(scope="module")
def run_patchlist(rtdata_module):
    rtdata = rtdata_module

    # This test should generate two match files
    args = [
        "r0000101001001001001_0002_wfi01_f158_cal.asdf",
        "r0000101001001001001_0002_wfi10_f158_cal.asdf",
    ]
    rtdata.get_data("WFI/image/r0000101001001001001_0002_wfi01_f158_cal.asdf")
    rtdata.get_data("WFI/image/r0000101001001001001_0002_wfi10_f158_cal.asdf")

    mk_skycell_list._cli(args)
    return rtdata


@pytest.mark.parametrize("expected_match_files", EXPECTED_MATCH_FILENAMES)
def test_match_files(run_patchlist, expected_match_files):
    # Test that the expected match files are created
    assert os.path.isfile(expected_match_files)


@pytest.fixture(scope="module")
def run_skycellasn(rtdata_module, run_patchlist):
    rtdata = rtdata_module

    # This test should generate several association files
    args = [
        "r0000101001001001001_0002_wfi01_f158_cal.match",
        "r0000101001001001001_0002_wfi10_f158_cal.match",
    ]

    mk_skycell_asn_from_skycell_list._cli(args)
    return rtdata


@pytest.mark.parametrize("expected_asn_files", EXPECTED_ASN_FILENAMES)
def test_asn_files(run_skycellasn, expected_asn_files):
    # Test that the expected asn files are created
    assert os.path.isfile(expected_asn_files)
