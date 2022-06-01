"""
Dark current subtraction
"""

import logging
import numpy as np


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def dark_subtraction(input_model, dark_ref_data):
    """
    Module for the dark current subtraction.

    Parameters
    ----------
    input_model : data model
        Input model assumed to be like a Roman RampModel.

    dark_model : data model
        Input model assumed to be like a Roman DarkRefModel.

    Returns
    -------
    output_data : ScienceData
        dark-subtracted science data
    """

    # Replace NaN's in the dark with zeros
    dark_ref_data.data[np.isnan(dark_ref_data.data)] = 0.0

    # Combine the dark and science DQ data
    input_model.pixeldq = np.bitwise_or(input_model.pixeldq, dark_ref_data.dq)

    # For Roman the shape of the dark reference files should match the
    # shape of the science data based on the MA table so we can do a direct
    # subtraction
    result = input_model.data - dark_ref_data.data
    input_model.data = result

    return input_model
