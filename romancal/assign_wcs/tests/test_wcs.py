from ast import Assign
import pytest

import numpy as np
from numpy.testing import assert_allclose

from gwcs.wcstools import grid_from_bounding_box

from romancal.assign_wcs.assign_wcs_step import AssignWcsStep
from roman_datamodels import datamodels as rdm
from roman_datamodels.testing import utils as testutil

from romancal.assign_wcs.utils import wcs_bbox_from_shape


def create_image():
    l2 = testutil.mk_level2_image()

    l2.meta.instrument.name = 'WFI'
    l2.meta.instrument.detector = 'WFI01'
    l2.meta.instrument.optical_element = 'F158'

    l2.meta.wcsinfo.v2_ref = -503
    l2.meta.wcsinfo.v3_ref = -318
    l2.meta.wcsinfo.ra_ref = 156
    l2.meta.wcsinfo.dec_ref = 54.2
    l2.meta.wcsinfo.vparity = -1
    l2.meta.wcsinfo.roll_ref = 0.15

    l2im = rdm.ImageModel(l2)
    return l2im


def create_distortion():
    distortions = [testutil.mk_distortion()]

    model = create_image()
    dist = testutil.mk_distortion()
    dist.coordinate_distortion_transform.bounding_box = wcs_bbox_from_shape(model.data.shape)
    distortions.append(dist)

    for distortion in distortions:
        distortion.meta['instrument']['name'] = 'WFI'
        distortion.meta['instrument']['detector'] = 'WFI01'
        distortion.meta['instrument']['optical_element'] = 'F158'

    return distortions


@pytest.mark.parametrize("distortion", create_distortion())
def test_wcs(distortion):
    l2im = create_image()
    l2_wcs = AssignWcsStep.call(l2im, override_distortion=distortion)

    assert l2_wcs.meta.wcs is not None
    assert l2_wcs.meta.cal_step.assign_wcs == 'COMPLETE'

    x, y = grid_from_bounding_box(l2_wcs.meta.wcs.bounding_box)

    ra, dec = l2_wcs.meta.wcs(x, y)
    assert_allclose(ra, l2im.meta.wcsinfo.ra_ref, atol=2.3)
    assert_allclose(dec, l2im.meta.wcsinfo.dec_ref, atol=1.3)

    # verify inputs outside the bounding box return NaNs
    ra, dec = l2_wcs.meta.wcs(-5, 3)
    assert np.isnan(ra)
    assert np.isnan(dec)
