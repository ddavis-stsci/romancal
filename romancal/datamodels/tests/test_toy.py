import pytest

import asdf
import numpy as np
from astropy.time import Time

from ..reference_files.flat import FlatModel
from ..reference_files.gain import GainModel

from romancal import datamodels
from stdatamodels.validate import ValidationWarning

# Define meta dictionary of required key / value pairs
FLATMETA = {"meta":
                {"instrument": {
                 "detector": "WFI01",
                 "optical_element": "F158",
                 "name": "WFI"
                },
                "telescope": "ROMAN",
                "model_type": "FlatModel",
                "date": Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc'),
                }
            }

# Testing reference_image_requirements schema
def test_valid_flat_schema(tmp_path):
    # Set temporary asdf file
    file_path = tmp_path / "test.asdf"

    # Testing flat asdf files
    with asdf.AsdfFile(FLATMETA) as af:
        # Test for valid entry
        af.write_to(file_path)

        with datamodels.FlatModel(file_path) as model:
            assert not model.validate()

def test_invalid_flat_schema(tmp_path):
    # Set temporary asdf file
    file_path = tmp_path / "test.asdf"

    # Testing flat asdf files
    with asdf.AsdfFile(FLATMETA) as af:
        # Test for invalid entry
        af['meta']['instrument'].pop("optical_element")
        af.write_to(file_path)

        with datamodels.FlatModel(file_path) as model:
            with pytest.warns(ValidationWarning):
                model.validate()

# Define meta dictionary of required key / value pairs
GAINMETA = {"meta":
                {"instrument": {
                 "detector": "WFI01",
                 "name": "WFI"
                },
                "telescope": "ROMAN",
                "model_type": "GainModel",
                "date": Time('1999-01-01T00:00:00.123456789', format='isot', scale='utc'),
                "calibration_software_version": "pi"
                }
            }

# Testing reference_image_requirements schema
def test_valid_gain_schema(tmp_path):
    # Set temporary asdf file
    file_path = tmp_path / "test.asdf"

    # Testing flat asdf files
    with asdf.AsdfFile(GAINMETA) as af:
        # Test for valid entry
        af.write_to(file_path)

        with datamodels.GainModel(file_path) as model:
            assert not model.validate()

def test_invalid_gain_schema(tmp_path):
    # Set temporary asdf file
    file_path = tmp_path / "test.asdf"

    # Testing flat asdf files
    with asdf.AsdfFile(GAINMETA) as af:
        # Test for invalid entry
        af['meta'].pop("calibration_software_version")
        af.write_to(file_path)

        with datamodels.GainModel(file_path) as model:
            with pytest.warns(ValidationWarning):
                model.validate()
