"""Roman tests for generating associatinos based on skycells"""

import json
import os
import pytest

from romancal.associations import mk_patchlist, mk_skycellasn

# mark all tests in this module
pytestmark = [pytest.mark.bigdata]

EXPECTED_MATCH_FILENAMES = [
    "r0000101001001001001_0002_wfi01_cal.match",
    "r0000101001001001001_0002_wfi10_cal.match",
]

EXPECTED_ASN_FILENAMES = [
    "r00001_p_v0000101001001001001_r274dp63x31y80_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x31y81_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x31y82_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x32y80_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x32y81_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x32y82_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x33y80_f158_asn.json",
    "r00001_p_v0000101001001001001_r274dp63x33y81_f158_asn.json",
]

@pytest.fixture(scope="module")
def run_patchlist(rtdata_module):
    rtdata = rtdata_module

    # This test should generate two match files
    args = [
        "r0000101001001001001_0002_wfi01_cal.asdf",
        "r0000101001001001001_0002_wfi10_cal.asdf",
    ]
    rtdata.get_data("WFI/image/r0000101001001001001_0002_wfi01_cal.asdf")
    rtdata.get_data("WFI/image/r0000101001001001001_0002_wfi10_cal.asdf")

    mk_patchlist._cli(args)
    return rtdata

@pytest.fixture(scope="module")
def output_patchlist(run_patchlist):
    return run_patchlist.output

@pytest.mark.parametrize("expected_match_files", EXPECTED_MATCH_FILENAMES)
def test_match_files(expected_match_files):
    # Test that the expected match files are created
    assert os.path.isfile(expected_match_files)

@pytest.fixture(scope="module")
def run_skycellasn(rtdata_module):
    rtdata = rtdata_module

    # This test should generate eight association files
    args = [
        "r0000101001001001001_0002_wfi01_cal.match",
        "r0000101001001001001_0002_wfi10_cal.match",
    ]
    rtdata.get_data("WFI/image/r0000101001001001001_0002_wfi01_cal.match")
    rtdata.get_data("WFI/image/r0000101001001001001_0002_wfi10_cal.match")

    mk_patchlist._cli(args)
    return rtdata

@pytest.fixture(scope="module")
def output_skycellasn(run_patchlist):
    return run_skycellasn.output

@pytest.mark.parametrize("expected_asn_files", EXPECTED_ASN_FILENAMES)
def test_asn_files(expected_asn_files):
    # Test that the expected match files are created
    assert os.path.isfile(expected_asn_files)
