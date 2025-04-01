"""
Utilities for fitting model PSFs to rate images.
"""

import logging
from collections import OrderedDict

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.table import Table
from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from photutils.psf import (
    GriddedPSFModel,
    IterativePSFPhotometry,
    PSFPhotometry,
    SourceGrouper,
)

__all__ = [
    "fit_psf_to_image_model",
    "get_psf_library",
]

# set loggers to debug level by default:
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Phase C central wavelengths [micron], released by Goddard (Jan 2023):
# https://roman.ipac.caltech.edu/sims/Param_db.html#wfi_filters
filter_central_wavelengths = {
    "F062": 0.620,
    "F087": 0.869,
    "F106": 1.060,
    "F129": 1.293,
    "F146": 1.464,
    "F158": 1.577,
    "F184": 1.842,
    "F213": 2.125,
}

default_finder = DAOStarFinder(
    # these defaults extracted from the
    # romancal SourceDetectionStep
    fwhm=1.0,
    threshold=0.0,
    sharplo=0.0,
    sharphi=1.0,
    roundlo=-1.0,
    roundhi=1.0,
    peakmax=None,
)


def get_psf_library(self):
    """Function to retrieve psf library from CRDS

    Compute a gridded PSF model for one SCA using the
    reference files in CRDS.
    The input reference files have 3 focus positions and this is using
    the in-focus images. There are also three spectral types that are
    available and this code uses the M5V spectal type.
    """
    # Open the reference file data model
    # select the infocus images (0) and we have a selection of spectral types
    # A0V, G2V, and M6V, pick M5V (2)
    focus = 0
    spectral_type = 2
    psf_images = self.psf_ref_model[0].psf[focus, spectral_type, :, :, :]
    # get the central position of the cutouts in a list
    psf_positions_x = self.psf_ref_model[0].meta.pixel_x.data.data
    psf_positions_y = self.psf_ref_model[0].meta.pixel_y.data.data
    meta = OrderedDict()
    position_list = []
    for index in range(len(psf_positions_x)):
        position_list.append([psf_positions_x[index], psf_positions_y[index]])

    meta["grid_xypos"] = position_list
    meta["oversampling"] = self.psf_ref_model[0].meta.oversample
    nd = NDData(psf_images, meta=meta)
    model = GriddedPSFModel(nd)

    return model


def fit_psf_to_image_model(
    image_model=None,
    data=None,
    error=None,
    mask=None,
    photometry_cls=PSFPhotometry,
    psf_model=None,
    grouper=None,
    fitter=None,
    localbkg_estimator=None,
    finder=None,
    x_init=None,
    y_init=None,
    progress_bar=False,
    error_lower_limit=None,
    fit_shape=(15, 15),
    exclude_out_of_bounds=True,
):
    """
    Fit PSF models to an ``ImageModel``.

    Parameters
    ----------
    image_model : `roman_datamodels.datamodels.ImageModel`
        Image datamodel. If ``image_model`` is supplied,
        ``data,error`` should be `None`.
    data : `astropy.units.Quantity`
        Fit a PSF model to the rate image ``data``.
        If ``data,error`` are supplied, ``image_model`` should be `None`.
    error : `astropy.units.Quantity`
        Uncertainties on fluxes in ``data``. Should be `None` if
        ``image_model`` is supplied.
    mask : 2D bool `numpy.ndarray`, optional
        Mask to apply to the data. Default is `None`.
    photometry_cls : {`photutils.psf.PSFPhotometry`,
            `photutils.psf.IterativePSFPhotometry`}
        Choose a photutils PSF photometry technique (default or iterative).
    psf_model : `astropy.modeling.Fittable2DModel`
        The 2D PSF model to fit to the rate image. Usually this model is an instance
        of `photutils.psf.GriddedPSFModel`.
    grouper : `photutils.psf.SourceGrouper`
        Specifies rules for attempting joint fits of multiple PSFs when
         there are nearby sources at small separations.
    fitter : `astropy.modeling.fitting.Fitter`, optional
        Modeling class which optimizes the PSF fit.
        Default is `astropy.modeling.fitting.LevMarLSQFitter(calc_uncertainties=True)`.
    localbkg_estimator : `photutils.background.LocalBackground`, optional
        Specifies inner and outer radii for computing flux background near
        a source. Default has ``inner_radius=10, outer_radius=30``.
    finder : subclass of `photutils.detection.StarFinderBase`, optional
        When ``photutils_cls`` is `photutils.psf.IterativePSFPhotometry`, the
        ``finder`` is called to determine if sources remain in the rate image
        after one PSF model is fit to the observations and removed.
        Default was extracted from the `DAOStarFinder` call in the
        Source Detection step.
    x_init : `numpy.ndarray`, optional
        Initial guesses for the ``x`` pixel coordinates of each source to fit.
    y_init : `numpy.ndarray`, optional
        Initial guesses for the ``y`` pixel coordinates of each source to fit.
    progress_bar : bool, optional
        Render a progress bar via photutils. Default is False.
    error_lower_limit : `astropy.units.Quantity`, optional
        Since some synthetic images may have bright sources with very
        small statistical uncertainties, the ``error`` can be clipped at
        ``error_lower_limit`` to prevent over-confident fits.
    fit_shape : int, or tuple of length 2, optional
        Rectangular shape around the center of a star that will
        be used to define the PSF-fitting data. See docs for
        `photutils.psf.PSFPhotometry` for details. Default is ``(16, 16)``.
    exclude_out_of_bounds : bool, optional
        If `True`, do not attempt to fit stars which have initial centroids
        that fall outside the pixel limits of the SCA. Default is False.

    Returns
    -------
    results_table : `astropy.table.QTable`
        PSF photometry results.
    photometry : instance of class ``photutils_cls``
        PSF photometry instance with configuration settings and results.

    """
    if grouper is None:
        # minimum separation before sources are fit simultaneously:
        grouper = SourceGrouper(min_separation=5)  # [pix]

    if fitter is None:
        fitter = LevMarLSQFitter(calc_uncertainties=True)

    # the iterative PSF method requires a finder:
    psf_photometry_kwargs = {}
    if photometry_cls is IterativePSFPhotometry or (x_init is None and y_init is None):
        if finder is None:
            finder = default_finder
        psf_photometry_kwargs["finder"] = finder

    if localbkg_estimator is None:
        localbkg_estimator = LocalBackground(
            inner_radius=10,  # [pix]
            outer_radius=30,  # [pix]
        )

    photometry = photometry_cls(
        grouper=grouper,
        localbkg_estimator=localbkg_estimator,
        psf_model=psf_model,
        fitter=fitter,
        fit_shape=fit_shape,
        aperture_radius=fit_shape[0],
        progress_bar=progress_bar,
        **psf_photometry_kwargs,
    )

    if x_init is not None and y_init is not None:
        guesses = Table(np.column_stack([x_init, y_init]), names=["x_init", "y_init"])
    else:
        guesses = None

    if image_model is None:
        if data is None and error is None:
            raise ValueError(
                "PSF fitting requires either an ImageModel, "
                "or arrays for the data and error."
            )

    if data is None and image_model is not None:
        data = image_model.data

    if error is None and image_model is not None:
        error = image_model.err

    if error_lower_limit is not None:
        # option to enforce a lower limit on the flux uncertainties
        error = np.clip(error, error_lower_limit, None)

    if exclude_out_of_bounds and guesses is not None:
        # don't attempt to fit PSFs for objects with initial centroids
        # outside the detector boundaries:
        init_centroid_in_range = (
            (guesses["x_init"] > 0)
            & (guesses["x_init"] < data.shape[1])
            & (guesses["y_init"] > 0)
            & (guesses["y_init"] < data.shape[0])
        )
        guesses = guesses[init_centroid_in_range]

    # fit the model PSF to the data:
    results_table = photometry(data=data, error=error, init_params=guesses, mask=mask)

    # results are stored on the PSFPhotometry instance:
    return results_table, photometry
