import os

import pytest
import roman_datamodels as rdm
from crds.core.exceptions import CrdsLookupError

from romancal.step import FlatFieldStep
from romancal.stpipe import RomanStep

from .regtestdata import compare_asdf


@pytest.mark.bigdata
def test_flat_field_image_step(rtdata, ignore_asdf_paths, resource_tracker, request):
    """Test for the flat field step using imaging data."""

    input_data = "r0000101001001001001_0001_wfi01_f158_assignwcs.asdf"
    rtdata.get_data(f"WFI/image/{input_data}")
    rtdata.input = input_data

    # Test CRDS
    step = FlatFieldStep()
    model = rdm.open(rtdata.input)
    ref_file_path = step.get_reference_file(model, "flat")
    ref_file_name = os.path.split(ref_file_path)[-1]
    assert "roman_wfi_flat" in ref_file_name
    # Test FlatFieldStep
    output = "r0000101001001001001_0001_wfi01_f158_flat.asdf"
    rtdata.output = output
    args = ["romancal.step.FlatFieldStep", rtdata.input]
    with resource_tracker.track(log=request):
        RomanStep.from_cmdline(args)

    rtdata.get_truth(f"truth/WFI/image/{output}")
    diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    assert diff.identical, diff.report()


@pytest.mark.bigdata
def test_flat_field_grism_step(rtdata, ignore_asdf_paths, resource_tracker, request):
    """Test for the flat field step using grism data. The reference file for
    the grism and prism data should be None, only testing the grism
    case here."""

    input_file = "r0000201001001001001_0001_wfi01_grism_assignwcs.asdf"
    rtdata.get_data(f"WFI/grism/{input_file}")
    rtdata.input = input_file

    # Test CRDS
    step = FlatFieldStep()
    model = rdm.open(rtdata.input)
    try:
        ref_file_path = step.get_reference_file(model, "flat")
        # Check for a valid reference file
        if ref_file_path != "N/A":
            ref_file_name = os.path.split(ref_file_path)[-1]
        elif ref_file_path == "N/A":
            ref_file_name = ref_file_path
    except CrdsLookupError:
        ref_file_name = None
    assert ref_file_name == "N/A"

    # Test FlatFieldStep
    output = "r0000201001001001001_0001_wfi01_grism_flat.asdf"
    rtdata.output = output
    args = ["romancal.step.FlatFieldStep", rtdata.input]
    with resource_tracker.track(log=request):
        RomanStep.from_cmdline(args)

    output_model = rdm.open(rtdata.output)
    assert output_model.meta.cal_step.flat_field == "SKIPPED"
    rtdata.get_truth(f"truth/WFI/grism/{output}")
    diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    assert diff.identical, diff.report()


@pytest.mark.bigdata
@pytest.mark.soctests
def test_flat_field_crds_match_image_step(
    rtdata, ignore_asdf_paths, resource_tracker, request
):
    """DMS79 Test: Testing that different datetimes pull different
    flat files and successfully make level 2 output"""

    # First file
    input_l2_file = "r0000101001001001001_0001_wfi01_f158_assignwcs.asdf"
    rtdata.get_data(f"WFI/image/{input_l2_file}")
    rtdata.input = input_l2_file

    # Test CRDS
    step = FlatFieldStep()
    model = rdm.open(rtdata.input)
    step.log.info(
        "DMS79 MSG: Testing retrieval of best ref file, Success is flat file with"
        " correct use after date"
    )

    step.log.info(f"DMS79 MSG: First data file: {rtdata.input.rsplit('/', 1)[1]}")
    step.log.info(f"DMS79 MSG: Observation date: {model.meta.exposure.start_time}")

    ref_file_path = step.get_reference_file(model, "flat")
    step.log.info(
        f"DMS79 MSG: CRDS matched flat file: {ref_file_path.rsplit('/', 1)[1]}"
    )
    flat = rdm.open(ref_file_path)
    step.log.info(f"DMS79 MSG: flat file UseAfter date: {flat.meta.useafter}")
    step.log.info(
        "DMS79 MSG: UseAfter date before observation date? :"
        f" {(flat.meta.useafter < model.meta.exposure.start_time)}"
    )

    # Test FlatFieldStep
    output = "r0000101001001001001_0001_wfi01_f158_flat.asdf"
    rtdata.output = output
    args = ["romancal.step.FlatFieldStep", rtdata.input]
    step.log.info(
        "DMS79 MSG: Running flat fielding step. The first ERROR is"
        "expected, due to extra CRDS parameters not having been "
        "implemented yet."
    )
    with resource_tracker.track(log=request):
        RomanStep.from_cmdline(args)

    rtdata.get_truth(f"truth/WFI/image/{output}")

    diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    step.log.info(
        f"DMS79 MSG: Was proper flat fielded Level 2 data produced? : {diff.identical}"
    )
    assert diff.identical, diff.report()

    # This test requires a second file, in order to meet the DMS79 requirement.
    # The test will show that two files with different observation dates match
    #  to separate flat files in CRDS.

    # Second file
    input_file = "r0000101001001001001_0001_wfi01_f158_changetime_assignwcs.asdf"
    rtdata.get_data(f"WFI/image/{input_file}")
    rtdata.input = input_file

    # Test CRDS
    step = FlatFieldStep()
    model = rdm.open(rtdata.input)

    step.log.info(f"DMS79 MSG: Second data file: {rtdata.input.rsplit('/', 1)[1]}")
    step.log.info(f"DMS79 MSG: Observation date: {model.meta.exposure.start_time}")

    ref_file_path_b = step.get_reference_file(model, "flat")
    step.log.info(
        f"DMS79 MSG: CRDS matched flat file: {ref_file_path_b.rsplit('/', 1)[1]}"
    )
    flat = rdm.open(ref_file_path_b)
    step.log.info(f"DMS79 MSG: flat file UseAfter date: {flat.meta.useafter}")
    step.log.info(
        "DMS79 MSG: UseAfter date before observation date? :"
        f" {(flat.meta.useafter < model.meta.exposure.start_time)}"
    )

    # Test FlatFieldStep
    output = "r0000101001001001001_0001_wfi01_f158_changetime_flat.asdf"
    rtdata.output = output
    args = ["romancal.step.FlatFieldStep", rtdata.input]
    step.log.info(
        "DMS79 MSG: Running flat fielding step. The first ERROR is"
        "expected, due to extra CRDS parameters not having been "
        "implemented yet."
    )
    RomanStep.from_cmdline(args)
    rtdata.get_truth(f"truth/WFI/image/{output}")
    diff = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    step.log.info(
        f"DMS79 MSG: Was proper flat fielded Level 2 data produced? : {diff.identical}"
    )
    assert diff.identical, diff.report()

    # Test differing flat matches
    step.log.info(
        "DMS79 MSG REQUIRED TEST: Are the two data files "
        "matched to different flat files? : "
        f"{('/'.join(ref_file_path.rsplit('/', 3)[1:]))} != "
        f"{('/'.join(ref_file_path_b.rsplit('/', 3)[1:]))}"
    )
    assert "/".join(ref_file_path.rsplit("/", 1)[1:]) != "/".join(
        ref_file_path_b.rsplit("/", 1)[1:]
    )
