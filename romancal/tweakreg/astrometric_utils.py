import os

import requests
from astropy import table
from astropy import units as u
from astropy.coordinates import SkyCoord

from ..assign_wcs import utils as wcsutil
from ..resample import resample_utils

ASTROMETRIC_CAT_ENVVAR = "ASTROMETRIC_CATALOG_URL"
DEF_CAT_URL = "http://gsss.stsci.edu/webservices"

# VO request timeout (in seconds)
TIMEOUT = 30.0

if ASTROMETRIC_CAT_ENVVAR in os.environ:
    SERVICELOCATION = os.environ[ASTROMETRIC_CAT_ENVVAR]
else:
    SERVICELOCATION = DEF_CAT_URL

"""

Primary function for creating an astrometric reference catalog.

"""


def create_astrometric_catalog(
    input_models,
    catalog="GAIADR3",
    output="ref_cat.ecsv",
    gaia_only=False,
    table_format="ascii.ecsv",
    existing_wcs=None,
    num_sources=None,
):
    """Create an astrometric catalog that covers the inputs' field-of-view.

    Parameters
    ----------
    input_models : str, list
        Filenames of images to be aligned to astrometric catalog

    catalog : str, optional
        Name of catalog to extract astrometric positions for sources in the
        input images' field-of-view. Default: GAIADR3. Options available are
        documented on the catalog web page.

    output : str, optional
        Filename to give to the astrometric catalog read in from the master
        catalog web service.  If None, no file will be written out.

    gaia_only : bool, optional
        Specify whether or not to only use sources from GAIA in output catalog

    table_format: str, optional
        Format to be used when writing the results to a file using the `output` option.
        A full list of the options can be found here:
        https://docs.astropy.org/en/stable/io/unified.html#built-in-readers-writers

    existing_wcs : model
        existing WCS object specified by the user as generated by
        `resample.resample_utils.make_output_wcs`

    num_sources : int
        Maximum number of brightest/faintest sources to return in catalog.
        If `num_sources` is negative, return that number of the faintest
        sources.  By default, all sources are returned.

    Notes
    -----
    This function will point to astrometric catalog web service defined
    through the use of the `ASTROMETRIC_CATALOG_URL` environment variable.
    Also, the default catalog to be used is set by the `DEF_CAT` variable.

    Returns
    -------
    ref_table : `~astropy.table.Table`
        Astropy Table object of the catalog

    """

    # start by creating a composite field-of-view for all inputs
    # This default output WCS will have the same plate-scale and orientation
    # as the first member in the list.
    # Fortunately, for alignment, this doesn't matter since no resampling of
    # data will be performed.
    if existing_wcs is not None:
        outwcs = existing_wcs
    else:
        outwcs = resample_utils.make_output_wcs(input_models)
    radius, fiducial = compute_radius(outwcs)

    # perform query for this field-of-view
    ref_dict = get_catalog(fiducial[0], fiducial[1], sr=radius, catalog=catalog)
    colnames = ("ra", "dec", "mag", "objID")

    ref_table = ref_dict[colnames]

    # Add catalog name as meta data
    ref_table.meta["catalog"] = catalog
    ref_table.meta["gaia_only"] = gaia_only

    # rename coordinate columns to be consistent with tweakwcs
    ref_table.rename_column("ra", "RA")
    ref_table.rename_column("dec", "DEC")

    # Append GAIA ID as a new column to the table...
    gaia_sources = []
    for source in ref_dict:
        if "GAIAsourceID" in source:
            g = source["GAIAsourceID"]
            if gaia_only and g.strip() == "":
                continue
        else:
            g = "-1"  # indicator for no source ID extracted
        gaia_sources.append(g)

    gaia_col = table.Column(data=gaia_sources, name="GaiaID", dtype="U25")
    ref_table.add_column(gaia_col)

    # sort table by magnitude, fainter to brightest
    ref_table.sort("mag", reverse=True)

    # If specified by the use through the 'num_sources' parameter,
    # trim the returned catalog down to just the brightest 'num_sources' sources
    # Should 'num_sources' be a negative value, it will return the faintest
    # 'num_sources' sources.
    if num_sources is not None:
        indx = -1 * num_sources
        ref_table = ref_table[:indx] if num_sources < 0 else ref_table[indx:]

    # Write out table to a file, if specified
    if output is not None:
        ref_table.write(output, format=table_format, overwrite=True)

    return ref_table


"""

Utility functions for creating an astrometric reference catalog.

"""


def compute_radius(wcs):
    """Compute the radius from the center to the furthest edge of the WCS."""

    fiducial = wcsutil.compute_fiducial([wcs], wcs.bounding_box)
    img_center = SkyCoord(ra=fiducial[0] * u.degree, dec=fiducial[1] * u.degree)
    wcs_foot = wcs.footprint()
    img_corners = SkyCoord(ra=wcs_foot[:, 0] * u.degree, dec=wcs_foot[:, 1] * u.degree)
    radius = img_center.separation(img_corners).max().value

    return radius, fiducial


def get_catalog(ra, dec, sr=0.1, catalog="GAIADR3"):
    """Extract catalog from VO web service.

    Parameters
    ----------
    ra : float
        Right Ascension (RA) of center of field-of-view (in decimal degrees)

    dec : float
        Declination (Dec) of center of field-of-view (in decimal degrees)

    sr : float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog.  Default: 0.1 degrees

    catalog : str, optional
        Name of catalog to query, as defined by web-service.  Default: 'GAIADR3'

    Returns
    -------
        A Table object of returned sources with all columns as provided by catalog.

    """
    service_type = "vo/CatalogSearch.aspx"
    spec_str = "RA={}&DEC={}&SR={}&FORMAT={}&CAT={}&MINDET=5"
    headers = {"Content-Type": "text/csv"}
    fmt = "CSV"

    spec = spec_str.format(ra, dec, sr, fmt, catalog)
    service_url = f"{SERVICELOCATION}/{service_type}?{spec}"
    rawcat = requests.get(service_url, headers=headers, timeout=TIMEOUT)
    r_contents = rawcat.content.decode()  # convert from bytes to a String
    rstr = r_contents.split("\r\n")
    # remove initial line describing the number of sources returned
    # CRITICAL to proper interpretation of CSV data
    del rstr[0]
    if len(rstr) == 0:
        print(Exception("VO catalog service returned no results."))
        raise

    return table.Table.read(rstr, format="csv")
