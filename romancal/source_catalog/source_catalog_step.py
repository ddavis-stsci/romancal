"""
Module for the source catalog step.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from astropy import coordinates
import astropy.units as u
import numpy as np
from astropy.table import join, QTable, Table
from photutils.segmentation import SegmentationImage
from roman_datamodels import datamodels
from roman_datamodels.datamodels import ImageModel
from roman_datamodels.dqflags import pixel

from romancal.datamodels.fileio import open_dataset
from romancal.source_catalog._background import RomanBackground
from romancal.source_catalog._detection import convolve_data, make_segmentation_image
from romancal.source_catalog._skyvals import compute_skyvals
from romancal.source_catalog._source_catalog import (
    DEFAULT_FLUX_UNIT,
    RomanSourceCatalog,
)
from romancal.source_catalog._utils import copy_model_arrays, get_ee_spline
from romancal.source_catalog.psf import add_jitter
from romancal.stpipe import RomanStep

if TYPE_CHECKING:
    from typing import ClassVar

__all__ = ["SourceCatalogStep"]

log = logging.getLogger(__name__)


class SourceCatalogStep(RomanStep):
    """
    Create a catalog of sources including photometry and basic shape
    measurements.

    Parameters
    ----------
    input : str, `ImageModel`, or `MosaicModel`
        Path to an ASDF file, or an `ImageModel` or `MosaicModel`.

    Other Parameters
    ----------------
    bkg_boxsize : int, optional
        Edge length, in pixels, of the square mesh boxes used by
        `~photutils.background.Background2D` to estimate the global 2D
        background.

    kernel_fwhm : float, optional
        Full-width-at-half-maximum, in pixels, of the Gaussian smoothing
        kernel used for source detection.

    snr_threshold : float, optional
        Per-pixel signal-to-noise ratio above the background required
        for a pixel to be considered part of a source.

    npixels : int, optional
        Minimum number of connected pixels above ``snr_threshold``
        required for a detection to be retained as a source.

    deblend : bool, optional
        If `True`, deblend overlapping sources after detection.

    suffix : str, optional
        Suffix appended to the output filenames.  Default ``'cat'``.

    fit_psf : bool, optional
        If `True`, fit source PSFs.

    forced_segmentation : str, optional
        If non-empty, path to a pre-computed segmentation image to use
        in place of fresh source detection (forced photometry mode).

    forced_photometry : str, optional
        If non-empty, path to a pre-computed source catalog to use
        in place of a computed source catalog (forced photometry mode).
    """

    class_alias = "source_catalog"

    reference_file_types: ClassVar = ["apcorr"]

    spec = """
        bkg_boxsize = integer(default=1000)   # background mesh box size in pixels
        kernel_fwhm = float(default=2.0)      # Gaussian kernel FWHM in pixels
        snr_threshold = float(default=3.0)    # per-pixel SNR threshold above the bkg
        npixels = integer(default=25)         # min number of pixels in source
        deblend = boolean(default=False)      # deblend sources?
        suffix = string(default='cat')        # Default suffix for output files
        fit_psf = boolean(default=True)       # fit source PSFs for accurate astrometry?
        forced_segmentation = string(default='')  # force the use of this segmentation map
        forced_photometry = string(default='') #  Input catalog name for forced photometry
        compute_skyvals = boolean(default=True)  # compute healpix sky summary arrays
    """

    def save_model(self, model, **kwargs):
        # depending on model set suffix and ext
        if isinstance(
            model,
            (
                datamodels.ForcedImageSourceCatalogModel,
                datamodels.ImageSourceCatalogModel,
                datamodels.ForcedMosaicSourceCatalogModel,
                datamodels.MosaicSourceCatalogModel,
                datamodels.TimeDomainSourceCatalogModel,
            ),
        ):
            kwargs["ext"] = "parquet"
            kwargs["suffix"] = kwargs.get("suffix", "cat")
        elif isinstance(
            model,
            (datamodels.SegmentationMapModel, datamodels.MosaicSegmentationMapModel),
        ):
            kwargs["suffix"] = kwargs.get("suffix", "segm")
        else:
            raise NotImplementedError(f"unsupported model: {type(model)}")

        # strip the index since these all have different extensions
        kwargs.pop("idx")

        return super().save_model(model, **kwargs)

    def _read_forced_detection_image(self, forced_segmodel):
        """
        Read the detection image from the forced segmentation model.

        Parameters
        ----------
        forced_segmodel : segmentation map model
            The open model given by ``forced_segmentation``.

        Returns
        -------
        result : `~numpy.ndarray` or `~astropy.units.Quantity`
            The detection image. It is a Quantity when its unit is
            known, in which case it is already in flux density units.
            Otherwise it is in the units of the image it was made from.

        Raises
        ------
        ValueError
            If the model has no ``detection_image`` array or its
            ``detection_image_unit`` is not a valid unit.
        """
        # Missing detection_image usually means an older file, an
        # empty/failed segmentation product, or a hand-built map.
        try:
            detection_image = forced_segmodel.detection_image
        except AttributeError as err:
            msg = (
                "forced_segmentation must include a detection_image array. "
                f"{self.forced_segmentation!r} does not; regenerate the "
                "segmentation map with SourceCatalogStep or "
                "MultibandCatalogStep, or provide a product that includes "
                "the original detection image."
            )
            raise ValueError(msg) from err

        # Use a plain ndarray view. The ``<<=`` operator silently leaves
        # an asdf NDArrayType without a unit.
        detection_image = np.asarray(detection_image)

        unit = getattr(forced_segmodel, "detection_image_unit", None)
        if unit is None and not isinstance(
            forced_segmodel, datamodels.MultibandSegmentationMapModel
        ):
            # Files written by SourceCatalogStep before the
            # detection_image_unit key was added hold a detection image
            # that was already converted to the catalog flux unit. Older
            # multiband files hold it in the units of the input images.
            unit = DEFAULT_FLUX_UNIT
            log.warning(
                "%r has no detection_image_unit. Assuming its "
                "detection_image is in %s.",
                self.forced_segmentation,
                unit,
            )

        if unit is not None:
            try:
                unit = u.Unit(unit)
            except ValueError as err:
                msg = (
                    f"The detection_image_unit {unit!r} in "
                    f"{self.forced_segmentation!r} is not a valid unit."
                )
                raise ValueError(msg) from err
            # This rebinds the name to a Quantity view of the array
            detection_image <<= unit

        return detection_image

    @staticmethod
    def _save_detection_image(segmentation_model, catobj):
        """
        Save the detection image used by a catalog and its unit.

        Parameters
        ----------
        segmentation_model : segmentation map model
            The output segmentation model.

        catobj : `RomanSourceCatalog`
            The catalog object. Its catalog must already be built
            because the detection image is converted to flux density
            units at that time.
        """
        detection_image = catobj.convolved_data
        segmentation_model["detection_image"] = detection_image.value
        segmentation_model["detection_image_unit"] = str(detection_image.unit)

    def _empty_catalog_results(
        self, cat_model, segmentation_model, input_model, model, mask
    ):
        """
        Build empty catalog and segmentation products.

        Empty products intentionally omit ``detection_image``. Forced
        photometry only needs that array when measuring sources.

        For ``ImageModel`` inputs the step still completed successfully,
        so ``meta.cal_step.source_catalog`` is marked ``COMPLETE``.
        """
        cat_model.source_catalog = cat_model.create_empty_catalog()
        segmentation_model.data = np.zeros(model.data.shape, dtype=np.uint32)
        self._attach_skyvals_if_enabled(input_model, segmentation_model, mask)
        # Match the non-empty path: only ImageModel records this cal_step.
        if isinstance(input_model, (datamodels.ImageModel | datamodels.MosaicModel)):
            self.finalize_result(input_model, self._reference_files_used)
            input_model.meta.cal_step.source_catalog = "COMPLETE"
        return cat_model, segmentation_model

    def _load_forced_segmentation(self, mask):
        """
        Load the forcing segmentation map and its detection image.

        Fully masked segments are removed. If no usable labels remain,
        both return values are `None` and ``detection_image`` is not
        required. Otherwise the forcing map must include
        ``detection_image``.

        Parameters
        ----------
        mask : 2D `~numpy.ndarray` of bool
            Bad-pixel mask for the image being cataloged.

        Returns
        -------
        segment_img : `~photutils.segmentation.SegmentationImage` or `None`
            The forcing segmentation, or `None` when empty.
        forced_detection_image : array-like or `None`
            Detection image from the forcing product, or `None` when
            the forcing map is empty.
        """
        with datamodels.open(self.forced_segmentation) as forced_segmodel:
            # forced_segmodel.data is asdf.tags.core.ndarray.NDArrayType
            forced_segimg = np.asarray(forced_segmodel.data[...]).copy()

            # Remove fully masked segments
            all_labels = np.unique(forced_segimg)
            unmasked_labels = np.unique(forced_segimg[~mask])
            fully_masked_labels = np.setdiff1d(
                all_labels, unmasked_labels, assume_unique=True
            )
            fully_masked_labels = fully_masked_labels[fully_masked_labels != 0]
            if fully_masked_labels.size > 0:
                forced_segimg[np.isin(forced_segimg, fully_masked_labels)] = 0

            # Empty forcing maps (e.g. zero-source prompt catalogs) do not
            # save detection_image. Produce an empty forced catalog instead
            # of requiring that array or measuring photometry on zero labels.
            if not np.any(forced_segimg):
                return None, None

            forced_detection_image = self._read_forced_detection_image(forced_segmodel)
            return SegmentationImage(forced_segimg), forced_detection_image

    @contextmanager
    def _open_psf_model(self, input_model):
        """Context to open the epsf reference file or None if fit_psf is not enabled."""
        if self.fit_psf:
            self.ref_file = self.get_reference_file(input_model, "epsf")
            log.info("Using ePSF reference file: %s", self.ref_file)
            with datamodels.open(self.ref_file) as psf_model:
                psf_model.psf = add_jitter(psf_model, input_model)
                yield psf_model
        else:
            yield None

    def _resolve_forced_centroids(self, src_table):
        if "x_centroid" in src_table.colnames:
            x_centroid = np.asarray(src_table["x_centroid"], dtype=np.float32)
        elif "x" in src_table.colnames:
            x_centroid = np.asarray(src_table["x"], dtype=np.float32)
            src_table["x_centroid"] = x_centroid
        else:
            raise ValueError("forced_photometry requires x or x_centroid column")

        if "y_centroid" in src_table.colnames:
            y_centroid = np.asarray(src_table["y_centroid"], dtype=np.float32)
        elif "y" in src_table.colnames:
            y_centroid = np.asarray(src_table["y"], dtype=np.float32)
            src_table["y_centroid"] = y_centroid
        else:
            raise ValueError("forced_photometry requires y or y_centroid column")

        return x_centroid, y_centroid

    def _build_forced_input_coord(self, input_model, src_table, x_centroid, y_centroid):
        input_coord = Table()
        input_coord["ra"] = np.full(len(src_table), np.nan, dtype=np.float64)
        input_coord["dec"] = np.full(len(src_table), np.nan, dtype=np.float64)
        input_coord["x_centroid"] = x_centroid.copy()
        input_coord["y_centroid"] = y_centroid.copy()

        if ("ra" in src_table.colnames) and ("dec" in src_table.colnames):
            ra = np.asarray(src_table["ra"], dtype=np.float64)
            dec = np.asarray(src_table["dec"], dtype=np.float64)
            input_coord["ra"] = ra
            input_coord["dec"] = dec
            for i, (ra_i, dec_i) in enumerate(zip(ra, dec, strict=False)):
                src_pos = coordinates.SkyCoord(ra=ra_i * u.deg, dec=dec_i * u.deg)
                outside = np.any(input_model.meta.wcs.outside_footprint([src_pos]))
                if outside:
                    x_pix, y_pix = input_model.meta.wcs.world_to_pixel(src_pos)
                    input_coord["x_centroid"][i] = np.float32(x_pix)
                    input_coord["y_centroid"][i] = np.float32(y_pix)

        return input_coord

    def _prepare_forced_photometry_inputs(self, input_model, model):
        model.meta.x_0_flag = True
        model.meta.y_0_flag = True
        model.meta.fwhm_flag = True

        try:
            src_table = Table.read(self.forced_photometry)
        except (FileNotFoundError, OSError) as err:
            msg = f"Cannot read forced_photometry catalog: {self.forced_photometry}"
            raise ValueError(msg) from err

        x_centroid, y_centroid = self._resolve_forced_centroids(src_table)
        input_coord = self._build_forced_input_coord(
            input_model, src_table, x_centroid, y_centroid
        )

        model.x_centroid = x_centroid.tolist()
        model.y_centroid = y_centroid.tolist()
        model.src_table = input_coord
        model.xycoords = (input_coord["x_centroid"], input_coord["y_centroid"])
        model.labels = list(range(0, len(model.x_centroid)))

        return src_table, input_coord

    def _build_forced_photometry_catalog(
        self,
        model,
        cat_model,
        detection_image,
        psf_model,
        mask,
        ee_spline,
        input_coord,
        src_table,
    ):
        model.meta.x_0_flag = False
        model.meta.y_0_flag = False
        model.meta.fwhm_flag = False

        floating_catobj = RomanSourceCatalog(
            model,
            cat_model,
            None,
            None,
            self.kernel_fwhm,
            fit_psf=self.fit_psf,
            psf_model=psf_model,
            mask=mask,
            cat_type="forced_photometry",
            ee_spline=ee_spline,
        )
        floating_cat = floating_catobj.catalog
        floating_cat.meta = None

        model.meta.x_0_flag = True
        model.meta.y_0_flag = True
        model.meta.fwhm_flag = True
        forced_catobj = RomanSourceCatalog(
            model,
            cat_model,
            None,
            detection_image,
            self.kernel_fwhm,
            fit_psf=self.fit_psf,
            psf_model=psf_model,
            mask=mask,
            cat_type="forced_photometry",
            ee_spline=ee_spline,
        )
        forced_cat = forced_catobj.catalog

        forced_cat.remove_columns(["id", "source_catalog_index", "forced_ra", "forced_dec"])
        forced_cat.remove_columns(["forced_image_flags", "forced_warning_flags"])
        cat = join(forced_cat, floating_cat, keys="label", join_type="outer")

        nrows = len(cat)
        cat["x"] = np.full(nrows, np.nan, dtype=np.float32) * u.pix
        cat["y"] = np.full(nrows, np.nan, dtype=np.float32) * u.pix
        cat["forced_psf_flags"] = np.zeros(nrows, dtype=np.int32)
        if len(input_coord) > 0:
            cat["x"] = input_coord["x_centroid"].astype(np.float32)
            cat["y"] = input_coord["y_centroid"].astype(np.float32)
        if "label" in src_table.colnames:
            cat["source_catalog_index"] = src_table["label"].astype(np.int16)

        return cat

    def _build_forced_detection_catalog(
        self,
        model,
        cat_model,
        segment_img,
        forced_detection_image,
        detection_image,
        psf_model,
        mask,
        ee_spline,
    ):
        forced_catobj = RomanSourceCatalog(
            model,
            cat_model,
            segment_img,
            forced_detection_image,
            self.kernel_fwhm,
            fit_psf=self.fit_psf,
            psf_model=psf_model,
            mask=mask,
            cat_type="forced_det",
            ee_spline=ee_spline,
        )
        forced_cat = forced_catobj.catalog

        model.meta.x_0_flag = False
        model.meta.y_0_flag = False
        model.meta.fwhm_flag = False
        floating_catobj = RomanSourceCatalog(
            model,
            cat_model,
            segment_img,
            detection_image,
            self.kernel_fwhm,
            fit_psf=self.fit_psf,
            psf_model=psf_model,
            mask=mask,
            cat_type="forced_full",
            ee_spline=ee_spline,
        )
        floating_cat = floating_catobj.catalog

        forced_cat.meta = None
        cat = join(floating_cat, forced_cat, keys="label", join_type="outer")
        return cat, forced_catobj

    def _build_prompt_catalog(
        self,
        model,
        cat_model,
        segment_img,
        detection_image,
        fit_psf,
        psf_model,
        mask,
        ee_spline,
    ):
        prompt_catobj = RomanSourceCatalog(
            model,
            cat_model,
            segment_img,
            detection_image,
            self.kernel_fwhm,
            fit_psf=fit_psf,
            psf_model=psf_model,
            mask=mask,
            cat_type="prompt",
            ee_spline=ee_spline,
        )
        return prompt_catobj.catalog, prompt_catobj

    @staticmethod
    def _assign_output_catalog(cat_model, cat_type, prompt_cat=None, cat=None):
        if cat_type == "prompt":
            cat_model.source_catalog = prompt_cat
        elif cat_type in ("forced_det", "forced_photometry"):
            cat_model.source_catalog = cat

    def _populate_segmentation_outputs(
        self,
        input_model,
        segmentation_model,
        segment_img,
        detection_image_catobj,
        mask,
    ):
        segmentation_model.data = segment_img.data.astype(np.uint32)
        self._save_detection_image(segmentation_model, detection_image_catobj)
        self._attach_skyvals_if_enabled(input_model, segmentation_model, mask)

    def process(self, dataset):
        input_model = open_dataset(dataset, update_version=self.update_version)

        # Define a boolean mask for pixels to be excluded
        mask = ~np.isfinite(input_model.data)
        mask |= ~np.isfinite(input_model.err)
        mask |= input_model.err <= 0

        # Copy the data and error arrays to avoid modifying the input model
        model = copy_model_arrays(input_model)

        # Create a DQ mask for ImageModel
        if isinstance(input_model, ImageModel):
            if model.dq.shape != model.data.shape:
                msg = (
                    f"model.dq shape {model.dq.shape} does not match "
                    f"model.data shape {model.data.shape}; expected a 2D "
                    "DQ array."
                )
                raise ValueError(msg)
            mask |= (model.dq & pixel.DO_NOT_USE) != 0

        # Initialize the source catalog model, copying the metadata
        # from the input model
        cat_model, segmentation_model = self._make_catalog_and_segmentation_models(
            model
        )

        # Return an empty segmentation image and catalog table if all
        # pixels are masked
        if np.all(mask):
            log.error("Cannot create source catalog. All pixels are masked.")
            return self._empty_catalog_results(
                cat_model, segmentation_model, input_model, model, mask
            )

        # In forced mode, resolve the forcing map before background and
        # convolution so empty maps return early without that work.
        forced_detection_image = None
        if self.forced_segmentation:
            segment_img, forced_detection_image = self._load_forced_segmentation(mask)
            if segment_img is None:
                log.info(
                    "Forced segmentation contains no usable sources; "
                    "returning an empty forced catalog."
                )
                return self._empty_catalog_results(
                    cat_model, segmentation_model, input_model, model, mask
                )

        # get the name of the psf reference file
        with self._open_psf_model(input_model) as psf_model:
            log.info("Calculating and subtracting background")
            bkg = RomanBackground(
                model.data,
                box_size=self.bkg_boxsize,
                coverage_mask=mask,
            )
            model.data -= bkg.background

            log.info("Creating detection image")
            detection_image = convolve_data(
                model.data, kernel_fwhm=self.kernel_fwhm, mask=mask
            )

            log.info("Detecting sources")
            if not self.forced_segmentation:
                segment_img = make_segmentation_image(
                    detection_image,
                    snr_threshold=self.snr_threshold,
                    n_pixels=self.npixels,
                    bkg_rms=bkg.background_rms,
                    deblend=self.deblend,
                    mask=mask,
                )

            log.info("Creating ee_fractions model")
            apcorr_ref = self.get_reference_file(input_model, "apcorr")
            ee_spline = get_ee_spline(input_model, apcorr_ref)

        # Return an empty segmentation image and catalog table if no
        # sources are detected
        if segment_img is None:
            log.error("Cannot create source catalog. No sources were detected.")
            cat_model.source_catalog = cat_model.create_empty_catalog()
            segmentation_model.data = np.zeros(model.data.shape, dtype=np.uint32)
            self._attach_skyvals_if_enabled(input_model, segmentation_model, mask)
            input_model.meta.cal_step.source_catalog = "COMPLETE"
            return cat_model, segmentation_model

        is_forced_segmentation = bool(self.forced_segmentation)
        is_forced_photometry = bool(self.forced_photometry)

        model.is_forced_photometry = is_forced_photometry

        src_table = None
        input_coord = None
        if is_forced_photometry:
            src_table, input_coord = self._prepare_forced_photometry_inputs(
                input_model, model
            )

        log.info("Creating source catalog")
        cat_type = "forced_det" if is_forced_segmentation else "prompt"
        fit_psf = self.fit_psf and (not is_forced_segmentation)  # skip when forced
        prompt_cat = None
        cat = None
        detection_image_catobj = None

        if cat_type == "prompt" and (not is_forced_photometry):
            prompt_cat, detection_image_catobj = self._build_prompt_catalog(
                model,
                cat_model,
                segment_img,
                detection_image,
                fit_psf,
                psf_model,
                mask,
                ee_spline,
            )
        elif is_forced_photometry:
            cat_type = "forced_photometry"
            cat = self._build_forced_photometry_catalog(
                model,
                cat_model,
                detection_image,
                psf_model,
                mask,
                ee_spline,
                input_coord,
                src_table,
            )
        elif cat_type == "forced_det":
            cat, detection_image_catobj = self._build_forced_detection_catalog(
                model,
                cat_model,
                segment_img,
                forced_detection_image,
                detection_image,
                psf_model,
                mask,
                ee_spline,
            )

        self._assign_output_catalog(cat_model, cat_type, prompt_cat=prompt_cat, cat=cat)

        log.info("Saving detection image")

        # Set the data and detection image
        if not model.is_forced_photometry:
            self._populate_segmentation_outputs(
                input_model,
                segmentation_model,
                segment_img,
                detection_image_catobj,
                mask,
            )
        # we update the input_model here to note that source_catalog finished
        # only for ImageModel as L3 doesn't have cal_step.source_catalog
        # and was not previously recorded
        if isinstance(input_model, datamodels.ImageModel):
            self.finalize_result(input_model, self._reference_files_used)
            input_model.meta.cal_step.source_catalog = "COMPLETE"
        if not is_forced_photometry:
            return cat_model, segmentation_model
        else:
            return cat_model

    def _make_catalog_and_segmentation_models(self, model):
        if isinstance(model, ImageModel):
            if self.forced_segmentation:
                cat_model_cls = datamodels.ForcedImageSourceCatalogModel
            elif self.forced_photometry:
                cat_model_cls = datamodels.TimeDomainSourceCatalogModel
            else:
                cat_model_cls = datamodels.ImageSourceCatalogModel
            segmentation_model_cls = datamodels.SegmentationMapModel
        else:
            if self.forced_segmentation:
                cat_model_cls = datamodels.ForcedMosaicSourceCatalogModel
            else:
                cat_model_cls = datamodels.MosaicSourceCatalogModel
            segmentation_model_cls = datamodels.MosaicSegmentationMapModel

        cat_model = cat_model_cls.create_minimal({"meta": model.meta})
        cat_model.meta["image"] = {
            "filename": model.meta.filename,
            "file_date": model.meta.file_date,
        }
        # copy over the data release id since there is no association input
        if "data_release_id" in model.meta:
            cat_model.meta.data_release_id = model.meta.data_release_id

        # make L3 metadata
        if self.forced_segmentation:
            cat_model.meta.image.forced_segmentation = self.forced_segmentation

        segmentation_model = segmentation_model_cls.create_minimal(
            {"meta": cat_model.meta}
        )

        return cat_model, segmentation_model

    def _attach_skyvals_if_enabled(self, input_model, segmentation_model, mask):
        if not (
            isinstance(input_model, datamodels.ImageModel) and self.compute_skyvals
        ):
            return

        skyvals, healpix11_cov = compute_skyvals(
            input_model=input_model,
            segmentation=segmentation_model.data,
            bad_pixel_mask=mask,
        )
        segmentation_model["skyvals"] = skyvals
        segmentation_model["healpix11_cov"] = healpix11_cov
