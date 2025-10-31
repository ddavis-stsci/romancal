"""
Module for the multiband mos source catalog step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import join
from astropy.convolution import convolve_fft
from photutils.psf.matching import (CosineBellWindow, create_matching_kernel)
from photutils.psf import ImagePSF
from roman_datamodels import datamodels
#from photutils.datasets import (make_model_image, make_random_models_table)


from romancal.datamodels import ModelLibrary
from romancal.multiband_catalog.background import subtract_background_library
from romancal.multiband_catalog.detection_image import make_detection_image
from romancal.multiband_catalog.utils import add_filter_to_colnames, prefix_matched, make_mask
from romancal.source_catalog.background import RomanBackground
from romancal.source_catalog.detection import make_segmentation_image
from romancal.source_catalog.save_utils import save_all_results, save_empty_results
from romancal.source_catalog.source_catalog import RomanSourceCatalog
from romancal.source_catalog.utils import get_ee_spline, copy_coadd_meta
from romancal.source_catalog.psf import create_l3_psf_model
from romancal.stpipe import RomanStep

if TYPE_CHECKING:
    from typing import ClassVar

__all__ = ["MultibandMosCatalog"]

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class MultibandMosCatalog(RomanStep):
    """
    Create a multiband catalog of sources including photometry and basic
    shape measurements.

    Parameters
    -----------
    input : str or `~romancal.datamodels.ModelLibrary`
        Path to an ASDF file or a `~romancal.datamodels.ModelLibrary`
        that contains `~roman_datamodels.datamodels.MosaicImageModel`
        models.
    """

    class_alias = "multiband_mos_catalog"
    reference_file_types: ClassVar = ["apcorr"]

    spec = """
        bkg_boxsize = integer(default=100)   # background mesh box size in pixels
        kernel_fwhms = float_list(default=None)  # Gaussian kernel FWHM in pixels
        snr_threshold = float(default=3.0)    # per-pixel SNR threshold above the bkg
        npixels = integer(default=25)         # min number of pixels in source
        deblend = boolean(default=False)      # deblend sources?
        suffix = string(default='cat')        # Default suffix for output files
        fit_psf = boolean(default=True)       # fit source PSFs for accurate astrometry?
    """

    def process(self, library):
        # All input MosaicImages in the ModelLibrary are assumed to have
        # the same shape and be pixel aligned.
        if isinstance(library, str):
            library = ModelLibrary(library)
        if not isinstance(library, ModelLibrary):
            raise TypeError("library input must be a ModelLibrary object")

        if library.asn['asn_type'] == 'MultiCat':
            reference_index = []
            science_index = []
            for product in library.asn['products']:
                for index, member in enumerate(product['members']):
                    if 'reference' in member['exptype']:
                        reference_index.append(index)
                    if 'science' in member['exptype']:
                        science_index.append(index)

        # make sure we have reference and science images.
        if not reference_index or not science_index:
            log.info("The multiband catalog step requires a reference and science image")
            return

        cat_obj = None

        with library:
            if len(reference_index) != 1:
                log.info("Only one reference model is allowed for the multiband catalog")
                return
            reference_model = library.borrow(reference_index[0])
            ref_lambda = int(reference_model.meta.instrument.optical_element[1:])
            library.shelve(reference_model, modify=False)

        # Initialize the source catalog model, copying the metadata
        # from the input model
        cat_model = datamodels.MultibandSourceCatalogModel.create_minimal(
            {"meta": reference_model.meta}
        )
        if isinstance(reference_model, datamodels.MosaicModel):
            copy_coadd_meta(reference_model, cat_model)
        else:
            cat_model.meta.optical_element = (
                reference_model.meta.instrument.optical_element
            )

        log.info("Creating ee_fractions model for first image")
        apcorr_ref = self.get_reference_file(reference_model, "apcorr")
        ee_spline = get_ee_spline(reference_model, apcorr_ref)

        # Define the output filename for the source catalog model
        try:
            cat_model.meta.filename = library.asn["products"][0]["name"]
        except (AttributeError, KeyError):
            cat_model.meta.filename = "multiband_catalog"

        log.info("Calculating and subtracting background")
        library = subtract_background_library(library, self.bkg_boxsize)

        log.info("Creating detection image")
        # Define the kernel FWHMs for the detection image
        # TODO: sensible defaults
        # TODO: redefine in terms of intrinsic FWHM
        if self.kernel_fwhms is None:
            self.kernel_fwhms = [2.0, 20.0]

        # TODO: det_img is saved in the MosaicSegmentationMapModel;
        # do we also want to save the det_err?
        det_img, det_err = make_detection_image(library, self.kernel_fwhms)

        # Estimate background rms from detection image to calculate a
        # threshold for source detection
        mask = ~np.isfinite(det_img) | ~np.isfinite(det_err) | (det_err <= 0)

        # Return an empty segmentation image and catalog table if all
        # pixels are masked in the detection image.
        if np.all(mask):
            msg = "Cannot create source catalog. All pixels in the detection image are masked."
            return save_empty_results(self, det_img.shape, cat_model, msg=msg)

        bkg = RomanBackground(
            det_img,
            box_size=self.bkg_boxsize,
            coverage_mask=mask,
        )
        bkg_rms = bkg.background_rms

        log.info("Detecting sources")
        segment_img = make_segmentation_image(
            det_img,
            snr_threshold=self.snr_threshold,
            npixels=self.npixels,
            bkg_rms=bkg_rms,
            deblend=self.deblend,
            mask=mask,
        )

        if segment_img is None:  # no sources found
            msg = "Cannot create source catalog. No sources were detected."
            return save_empty_results(self, det_img.shape, cat_model, msg=msg)

        segment_img.detection_image = det_img

        # Define the detection image model
        det_model = datamodels.MosaicModel()
        det_model.data = det_img
        det_model.err = det_err

        # TODO: this is a temporary solution to get model attributes
        # currently needed in RomanSourceCatalog
        det_model.weight = reference_model.weight
        det_model.meta = reference_model.meta

        # The stellar FWHM is needed to define the kernel used for
        # the DAOStarFinder sharpness and roundness properties.
        # TODO: measure on a secondary detection image with minimal
        # smoothing?; use the same detection image for basic shape
        # measurements?
        star_kernel_fwhm = np.min(self.kernel_fwhms)

        log.info("Creating catalog for detection image")
        det_catobj = RomanSourceCatalog(
            det_model,
            segment_img,
            det_img,
            star_kernel_fwhm,
            fit_psf=self.fit_psf,
            detection_cat=None,
            mask=mask,
            cat_type="dr_det",
            ee_spline=ee_spline,
        )

        # Generate the catalog for the detection image.
        # We need to make this catalog before we pass det_catobj
        # to the RomanSourceCatalog constructor.
        det_cat = det_catobj.catalog

        if self.fit_psf:
            # Create catalogs
            # If cat_obj is None create the reference catalog
            if cat_obj is None:
                # get the reference image
                #pdb.set_trace()
                model = reference_model
                #reference_image_data = model.copy()
                reference_image_data = model.data

                mask_ref = make_mask(reference_model)

                psf_ref_file = self.get_reference_file(reference_model, "epsf")
                psf_reference_model = datamodels.open(psf_ref_file)
                ref_filter_name = model.meta.instrument.optical_element
                log.info(f"Creating catalog for {ref_filter_name} image")
                apcorr_ref = self.get_reference_file(model, "apcorr")
                ee_spline = get_ee_spline(model, apcorr_ref)

                # Make the catalog for the reference image
                catobj = RomanSourceCatalog(
                      reference_model,
                      segment_img,
                      None,
                      star_kernel_fwhm,
                      fit_psf=self.fit_psf,
                      detection_cat=det_catobj,
                      mask=mask_ref,
                      psf_ref_model=psf_reference_model,
                      cat_type="dr_band",
                      ee_spline=ee_spline,
                  )
                 # Add the filter name to the column names
                cat = add_filter_to_colnames(catobj.catalog, ref_filter_name)
                ee_fractions = cat.meta["ee_fractions"]
                det_cat = join(det_cat, cat, keys="label", join_type="outer")

            # now process the science images based on the relative wavelength
            for idx, _ in enumerate(science_index):
                science_file_name = library.asn['products'][0]['members'][science_index[idx]]['expname']
                science_image_model = library._loaded_models[science_index[idx]]
                log.info(f"Processing for science image {science_file_name} {idx}")
                science_filter_name = science_image_model.meta.instrument.optical_element
                det_cat.meta[f"ee_fractions_{science_filter_name.lower()}"] = {}
                # set the windowing function for create_matching_kernel for the reference image
                # and the match_image
                window = CosineBellWindow(alpha=0.35)
                threshold = 0.15
                # set the oversample parameter for creating the l3 psf model
                oversample = 11
                # generate L3 psf's for the reference image in the association
                _, l3_psf_reference_image= create_l3_psf_model(
                    psf_reference_model,
                    pixel_scale=reference_model.meta.wcsinfo.pixel_scale
                    * 3600.0 ,  # wcsinfo is in degrees. Need arcsec
                    pixfrac=reference_model.meta.resample.pixfrac,
                    oversample=oversample,
                    threshold=threshold,
                )

                # information and CRDS epsf ref files for the matching  science image
                science_image_lambda = int(science_image_model.meta.instrument.optical_element[1:])
                psf_science_file = self.get_reference_file(science_image_model, "epsf")
                psf_science_model = datamodels.open(psf_science_file)

                # trim the l3_psf_image
                # the returned L3 PSF image contains many low values far in the wings that supply noise when the
                # FFT smoothing is applied. This supresses that source of noise but should have a more robust
                # fix later
                l3_psf_reference_image[l3_psf_reference_image <
                                       np.average(l3_psf_reference_image[:,0:260])*950.] = 0.
                # setup mask for the input data
                mask = make_mask(science_image_model)

                log.info(f" Model lambda: {science_image_lambda}, Ref lambda {ref_lambda}")
                if science_image_lambda < ref_lambda:
                    # setup a matching psf model for the non-reference model based on the reference psf
                    # convolve the match data with the level 3 reference image psf
                    log.info(f"Start the convolution for science index {idx}, lambda={science_image_lambda}")
                    # Trim the science data, currently processing results in large negative pixels which affect
                    # the convolution in unphysical ways
                    science_image_model.data[science_image_model.data<0.]=0.
                    # generate L3 psf's for the science image to be matched to the reference image
                    l3_psf_match_model,  l3_psf_match_image = create_l3_psf_model(
                        psf_science_model,
                        pixel_scale=science_image_model.meta.wcsinfo.pixel_scale
                                     * 3600.0,  # wcsinfo is in degrees. Need arcsec
                        pixfrac=science_image_model.meta.resample.pixfrac,
                        oversample= oversample,
                        threshold=threshold,
                    )
                    np.save(f'l3_psf_matched_{science_image_lambda}_image', l3_psf_match_image)
                    # trim the l3_psf_image
                    # the returned L3 PSF image contains many low values far in the wings that supply noise when the
                    # FFT smoothing is applied. This supresses that source of noise. 
                    l3_psf_reference_image[l3_psf_reference_image <
                                           np.average(l3_psf_reference_image[:,0:260])*950.] = 0. 

                    # Convolve the science data to match the PSF of the reference image
                    data_conv = convolve_fft(
                        science_image_model.data,
                        l3_psf_reference_image,
                        mask=mask,
                        preserve_nan=False,
                        nan_treatment='interpolate',
                        normalize_kernel=True)
                    #log.info(f"total match data before convolution {np.nansum(science_image_model.data)}")
                    #log.info(f"total match data after convolution {np.nansum(data_conv)}")
                    log.info("Create matched psf")
                    # Match the match image psf to the reference image psf
                    matched_l3_psf = create_matching_kernel(l3_psf_match_image,
                                                            l3_psf_reference_image,
                                                            window=window)
                    
                    # Use PSF utils ImagePSF to create a fitable model for photutils
                    x_0, y_0 = matched_l3_psf.shape
                    x_0 = (x_0 - 1) / 2.0 / oversample
                    y_0 = (y_0 - 1) / 2.0 / oversample
                    matched_l3_model = ImagePSF(matched_l3_psf,
                                                x_0=x_0, y_0=y_0,
                                                oversampling=oversample)
                    np.save(f'l3_psf_matched_{science_image_lambda}_image', l3_psf_match_image)

                    apcorr_ref = self.get_reference_file(science_image_model, "apcorr")
                    ee_spline = get_ee_spline(science_image_model, apcorr_ref)
                    log.info("Create the src catalog")
                    # replace the science data with the convolved data
                    science_image_model.data = data_conv
                    log.info(f"Creating catalog for {science_filter_name} image")
                    catobj = RomanSourceCatalog(
                        science_image_model,
                        segment_img,
                        None,
                        star_kernel_fwhm,
                        fit_psf=self.fit_psf,
                        detection_cat=det_catobj,
                        mask=mask,
                        psf_ref_model=matched_l3_model,
                        cat_type="dr_band",
                        ee_spline=ee_spline,
                    )
                    log.info(f"Adding {science_filter_name} to columns")
                    catobj.catalog = add_filter_to_colnames(catobj.catalog, science_filter_name)
                    cat = prefix_matched(catobj)
                    ee_fractions = [f"ee_fractions_{science_filter_name.lower()}"]
                    det_cat = join(det_cat, cat, keys="label", join_type="outer")
                    det_cat.meta[f"ee_fractions_{science_filter_name.lower()}"] = ee_fractions

                elif science_image_lambda > ref_lambda:
                    # setup a matching psf model for the non-reference model based on the reference psf
                    # convolve the reference data with the level 3  match image psf
                    log.info(f"Start processing science image with lambda= {science_image_lambda}")
                    log.info("Create matched psf for science data redder that reference")

                    _, l3_psf_science_image= create_l3_psf_model(
                        psf_science_model,
                        pixel_scale=reference_model.meta.wcsinfo.pixel_scale
                        * 3600.0 ,  # wcsinfo is in degrees. Need arcsec
                        #pixfrac=reference_model.meta.resample.pixfrac,
                        pixfrac=1.0,
                        oversample=oversample,
                        threshold=threshold,
                    )
                    # match the reference psf to the science image psf
                    matched_l3_psf = create_matching_kernel(l3_psf_reference_image,
                                                            l3_psf_science_image,
                                                            window=window)
                    np.save(f'l3_psf_science_{science_image_lambda}_image', l3_psf_science_image)
                    np.save(f'l3_psf_reference_{ref_lambda}_image', l3_psf_reference_image)
                    np.save(f'l3_matched_psf_{science_image_lambda}_image', matched_l3_psf)

                    #x_0, y_0 = matched_l3_psf.shape
                    #x_0 = (x_0 - 1) / 2.0 / oversample
                    #y_0 = (y_0 - 1) / 2.0 / oversample

                    # Generate a psf model for the redder science image to convolve the
                    # refererence image so the ref psf matches the science image psf
                    #matched_l3_model = ImagePSF(matched_l3_psf,  x_0=x_0, y_0=y_0, oversampling=oversample)
                    _, l3_psf_match_image= create_l3_psf_model(
                        psf_science_model,
                        pixel_scale=reference_model.meta.wcsinfo.pixel_scale
                        * 3600.0 ,  # wcsinfo is in degrees. Need arcsec
                        pixfrac=reference_model.meta.resample.pixfrac,
                        oversample=oversample,
                        threshold=threshold,
                    )

                    # Diagnostics
                    # trim the l3_psf_image
                    # the returned L3 PSF image contains many low values far in the wings that supply noise when the
                    # FFT smoothing is applied. This supresses that source of noise. 
                    #l3_psf_reference_image[l3_psf_reference_image < np.average(l3_psf_reference_image[:,0:260])*950.] = 0. 
                    #n_sources = 1
                    #shape = (885, 885)
                    #param_ranges = {'amplitude': [1.00, 1.01], 'x_0': [442, 443], 'y_0': [442, 443]}
                    #params = make_random_models_table(n_sources, param_ranges, seed=0)
                    #model_shape = (15, 15)
                    #l3_psf_match_image = make_model_image(shape, l3_psf_reference_model, params, model_shape=model_shape)
                    #l3_psf_match_image = make_model_image(shape, l3_psf_match_model, params, model_shape=model_shape)
                    #log.info(f'Writing science image psf, l3_psf_matched_{science_image_lambda}_image')
                    #np.save(f'l3_psf_matched_{science_image_lambda}_image', l3_psf_match_image)
                    #log.info(f'Writing reference image psf, l3_psf_matched_{science_image_lambda}_image')
                    #np.save(f'l3_psf_matched_{ref_lambda}_image', l3_psf_reference_image)
                    #pdb.set_trace()

                    # Convolved the reference image with the science image psf
                    data_conv = convolve_fft(
                        reference_model.data,
                        l3_psf_match_image,
                        mask=mask_ref,
                        preserve_nan=False,
                        nan_treatment='interpolate',
                        normalize_kernel=True)
                    #log.info(f"total match data before convolution {np.nansum(science_image_model.data)}")
                    #log.info(f"total match data after convolution {(np.nansum(science_image_model.data)-np.nansum(data_conv>0.))/np.nansum(data_conv>0.)}")
                    #log.info(f"%diff  after convolution {np.nansum(data_conv)}")
                    apcorr_ref = self.get_reference_file(science_image_model, "apcorr")
                    ee_spline = get_ee_spline(science_image_model, apcorr_ref)
                    # Clean up the FFT smoothed image to prevent crazy fluxes, e.g. negative or inifinite values
                    reference_model.data = np.nan_to_num(data_conv, copy=True, nan=0.0)
                    log.info(f"Creating catalog for {science_filter_name} image")
                    # Run the catalog on the match image
                    science_image_optical_element = science_image_model.meta.instrument.optical_element
                    mask_match = make_mask(science_image_model)

                    # Generate the photometry catalog for the science image
                    catobj = RomanSourceCatalog(
                        science_image_model,
                        segment_img,
                        None,
                        star_kernel_fwhm,
                        fit_psf=self.fit_psf,
                        detection_cat=det_catobj,
                        mask=mask_match,
                        #psf_ref_model=matched_psf_model,
                        psf_ref_model=matched_l3_model,
                        cat_type="dr_band",
                        ee_spline=ee_spline,
                    )
                    log.info(f"Adding {science_image_optical_element} to columns")
                    #add ref filter so that columns for redder image matching do not colide
                    catobj.catalog = add_filter_to_colnames(catobj.catalog, science_image_optical_element)
                    det_cat = join(det_cat, catobj.catalog, keys="label", join_type="outer")

                    # Run the catalog on the convolved reference image
                    #science_filter_name_ref = reference_image_model.meta.instrument.optical_element
                    mask_ref = make_mask(reference_model)
                    reference_model.data = data_conv
                    # rerun the catalog on the matched smoothed reference image data
                    catobj = RomanSourceCatalog(
                        reference_model,
                        segment_img,
                        None,
                        star_kernel_fwhm,
                        fit_psf=self.fit_psf,
                        detection_cat=det_catobj,
                        mask=mask_ref,
                        psf_ref_model=matched_l3_model,
                        cat_type="dr_band",
                        ee_spline=ee_spline,
                    )

                    # Replace the reference image data with the unconvolved data
                    reference_model.data = reference_image_data
                    # save the matched (convolved) ref data to matched_{ref_lambda}_{match_lambda}
                    filter_name_match = ref_filter_name+"_"+science_image_optical_element
                    log.info(f"Adding {filter_name_match} to columns")
                    catobj.catalog = add_filter_to_colnames(catobj.catalog, filter_name_match)
                    cat = prefix_matched(catobj)
                    match_band = f'matched_kron_{filter_name_match}_flux'.lower()
                    ref_band = f'kron_{ref_filter_name}_flux'.lower()
                    log.info(f"Adding {match_band} to the catalog")
                    det_cat = join(det_cat, cat, keys="label", join_type="outer")           
                    log.info(f'Adding correction column C_{filter_name_match} to the catalog')
                    det_cat.add_column(det_cat[match_band]/det_cat[ref_band],
                                       name=f'C_{filter_name_match}')
                    # find the columns to correct
                    cols_to_correct = [x for x in det_cat.colnames
                                       if "_flux" in x if f"f+{ref_lambda}" not in
                                       x if f"f{science_image_lambda}" in x
                                       if "matched" not in x]
                    for column_name in cols_to_correct:
                        log.info(f'Adding corrected flux column matched_{column_name} to the catalog')
                        det_cat.add_column(det_cat[column_name]*det_cat[f'C_{filter_name_match}'],
                                           name='matched_'+column_name)
            apcorr_ref = self.get_reference_file(model, "apcorr")
            ee_spline = get_ee_spline(model, apcorr_ref)

        # TODO: what metadata do we want to keep, if any,
        # from the filter catalogs?
        cat.meta = None

        # Put the resulting multiband catalog in the model
        cat_model.source_catalog = det_cat

        return save_all_results(self, segment_img, cat_model, input_model=reference_model)
