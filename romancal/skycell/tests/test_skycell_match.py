"""
Unit tests for skycell.match.

These tests depend very strongly on the contents of the referenced table of patches.
Changes to the contents of this table will require changes to the tests for
any unit tests that depend on specific matches to patches in the table.

Any changes to the matching algorithm should be completely separate from any
changes to the table contents.

The tests include the following cases to validate the matches
1) Simple case at ra, dec, pa = 0, 0, 0
2) 1) translated to ra, de, pa = 180, 40, 0
3) Same as 2) but with pa = 45 and 60.
4) Same as 3) (pa=45 case) but with the lower corner just above, below, to the right
   and to the left of a 4 patch corner
   (assuming non-overlapping patchs within a common tangent point).
   This requires identifying the ra, dec of such a corner in the table.
5) A test of a WCS provided example.

Most of these tests check to see if the matches are what are expected, by index of
the table (the table format is expected to include the index of the entry as one
of its columns so that subsets of the table selected to reduce the filesize still
retain the same index obtained.)
"""

from pathlib import Path

import astropy.coordinates as coord
import astropy.modeling.models as amm
import astropy.units as u
import numpy as np
import pytest
import spherical_geometry.vector as sgv
from gwcs import WCS, coordinate_frames

import romancal.skycell.match as sm
from romancal.skycell import skymap

TEST_POINTS = [
    (0.88955854, 87.53857137),
    (20.6543883, 87.60498618),
    (343.19474696, 85.05565535),
    (8.94286202, 85.50465173),
    (27.38417684, 85.03404907),
    (310.53503934, 88.56749324),
]
EPSILON = 0.0011  # epsilon offset in degrees
DATA_DIRECTORY = Path(__file__).parent / "data"


@pytest.fixture()
def skymap_subset() -> skymap.SkyMap:
    """
    smaller subset to allow these tests
    to run without access to the full skymap from CRDS.
    """
    return skymap.SkyMap(DATA_DIRECTORY / "skymap_subset.asdf")


def mk_im_corners(
    ra: float, dec: float, pa: float, size: float
) -> list[tuple[float, float]]:
    """
    Generate 4 image corners of a square with the center at the supplied
    side size, ra, dec, and position angle (all in degrees).
    """
    # Generate 4 unit vectors at ra, dec = (0 , 0)
    center = sgv.lonlat_to_vector(0.0, 0.0)
    radecvec = sgv.lonlat_to_vector(ra, dec)
    zaxis = (0.0, 0.0, 1.0)
    yaxis = (0.0, 1.0, 0.0)
    pp = sgv.rotate_around(
        *(sgv.rotate_around(*(center + yaxis + (-size / 2,))) + zaxis + (+size / 2,))
    )
    pm = sgv.rotate_around(
        *(sgv.rotate_around(*(center + yaxis + (+size / 2,))) + zaxis + (+size / 2,))
    )
    mp = sgv.rotate_around(
        *(sgv.rotate_around(*(center + yaxis + (-size / 2,))) + zaxis + (-size / 2,))
    )
    mm = sgv.rotate_around(
        *(sgv.rotate_around(*(center + yaxis + (+size / 2,))) + zaxis + (-size / 2,))
    )
    rect = [pp, mp, mm, pm]

    # Now move to requested ra and dec
    trect = [
        sgv.rotate_around(
            *(sgv.rotate_around(*(vec + yaxis + (-dec,))) + zaxis + (ra,))
        )
        for vec in rect
    ]
    # Rotate to desired position angle
    rrect = [sgv.rotate_around(*(vec + radecvec + (pa,))) for vec in trect]
    frect = [sgv.vector_to_lonlat(*vec) for vec in rrect]
    # Reorganize by ra, dec arrays
    radecrect = np.array(frect)
    return radecrect


def mk_gwcs(ra, dec, pa, bounding_box=None) -> WCS:
    """
    Construct a GWCS model for testing the patch matching when provided a WCS
    This just implements a basic tangent projection with specified ra, dec, and
    position angle
    """
    transform = (amm.Shift(-2048) & amm.Shift(-2048)) | (
        amm.Scale(0.11 / 3600.0) & amm.Scale(0.11 / 3600.0)
        | amm.Rotation2D(pa)
        | amm.Pix2Sky_TAN()
        | amm.RotateNative2Celestial(ra, dec, 180.0)
    )
    detector_frame = coordinate_frames.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky_frame = coordinate_frames.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )
    wcsobj = WCS([(detector_frame, transform), (sky_frame, None)])
    if bounding_box is not None:
        wcsobj.bounding_box = bounding_box
        wcsobj.array_shape = tuple(
            int(bounding_box[index][1] - bounding_box[index][0])
            for index in range(len(bounding_box))
        )
    return wcsobj


@pytest.mark.parametrize(
    "test_point,offset,rotation,size,expected_skycell_names",
    [
        (
            TEST_POINTS[0],
            (0, 0),
            45,
            0.001,
            (
                "000p86x66y51",
                "000p86x66y50",
                "000p86x65y51",
                "000p86x65y50",
            ),
        ),
        (
            TEST_POINTS[0],
            (0, +EPSILON),
            45,
            0.001,
            (
                "000p86x66y50",
                "000p86x66y51",
            ),
        ),
        (
            TEST_POINTS[1],
            (0, +EPSILON),
            45,
            0.001,
            (
                "000p86x69y61",
                "000p86x69y62",
            ),
        ),
        (
            TEST_POINTS[1],
            (0, -EPSILON),
            45,
            0.001,
            ("000p86x69y62", "000p86x68y62", "000p86x69y61", "000p86x68y61"),
        ),
        (
            TEST_POINTS[1],
            (+EPSILON, 0),
            45,
            0.001,
            ("000p86x69y62", "000p86x68y62", "000p86x69y61", "000p86x68y61"),
        ),
        (
            TEST_POINTS[1],
            (-EPSILON, 0),
            45,
            0.001,
            ("000p86x69y62", "000p86x69y61", "000p86x68y62", "000p86x68y61"),
        ),
        (
            TEST_POINTS[1],
            (0, 0),
            45,
            0.001,
            ("000p86x69y62", "000p86x68y62", "000p86x69y61", "000p86x68y61"),
        ),
        (
            TEST_POINTS[0],
            (0, 0),
            45,
            0.3,
            (
                "000p86x66y51",
                "000p86x66y50",
                "000p86x65y51",
                "000p86x65y50",
                "000p86x67y51",
                "000p86x66y52",
                "000p86x67y50",
                "000p86x65y52",
                "000p86x64y51",
                "000p86x66y49",
                "000p86x64y50",
                "000p86x65y49",
                "000p86x67y52",
                "000p86x67y49",
                "000p86x64y52",
                "000p86x64y49",
                "000p86x68y51",
                "000p86x66y53",
                "000p86x68y50",
                "000p86x65y53",
                "000p86x63y51",
                "000p86x66y48",
                "000p86x63y50",
                "000p86x65y48",
            ),
        ),
        (
            TEST_POINTS[1],
            (0, 0),
            45,
            0.5,
            (
                "000p86x69y62",
                "000p86x68y62",
                "000p86x69y61",
                "000p86x68y61",
                "000p86x70y62",
                "000p86x69y63",
                "000p86x70y61",
                "000p86x68y63",
                "000p86x67y62",
                "000p86x69y60",
                "000p86x67y61",
                "000p86x68y60",
                "000p86x70y60",
                "000p86x67y63",
                "000p86x67y60",
                "000p86x71y62",
                "000p86x71y61",
                "000p86x66y62",
                "000p86x69y59",
                "000p86x66y61",
                "000p86x68y59",
                "000p86x71y60",
                "000p86x66y63",
                "000p86x70y59",
                "000p86x66y60",
                "000p86x67y59",
                "000p86x72y61",
                "000p86x71y59",
                "000p86x66y64",
                "000p86x65y62",
                "000p86x69y58",
                "000p86x65y61",
                "000p86x68y58",
                "000p86x66y59",
                "000p86x72y60",
                "000p86x65y63",
                "000p86x70y58",
                "000p86x65y60",
                "000p86x67y58",
                "000p86x72y59",
                "000p86x65y64",
                "000p86x66y58",
                "000p86x73y61",
                "000p86x64y62",
                "000p86x68y57",
                "000p86x73y60",
                "000p86x64y63",
                "000p86x67y57",
            ),
        ),
        (
            TEST_POINTS[2],
            (0, 0),
            0,
            0.4,
            (
                "000p86x35y31",
                "000p86x35y30",
                "000p86x34y31",
                "000p86x34y30",
                "000p86x35y32",
                "000p86x36y31",
                "000p86x34y32",
                "000p86x36y30",
                "000p86x35y29",
                "000p86x33y31",
                "000p86x34y29",
                "000p86x33y30",
                "000p86x36y32",
                "000p86x36y29",
                "000p86x33y32",
                "000p86x33y29",
                "000p86x35y33",
                "000p86x37y31",
                "000p86x34y33",
                "000p86x37y30",
                "000p86x35y28",
                "000p86x32y31",
                "000p86x34y28",
                "000p86x32y30",
                "000p86x36y33",
                "000p86x37y32",
                "000p86x33y33",
                "000p86x37y29",
                "000p86x36y28",
                "000p86x32y32",
                "000p86x33y28",
                "000p86x32y29",
                "000p86x37y33",
                "000p86x35y34",
                "000p86x38y30",
                "000p86x37y28",
                "000p86x32y33",
                "000p86x31y31",
                "000p86x34y27",
                "000p86x32y28",
                "000p86x36y34",
                "000p86x38y29",
                "000p86x31y32",
                "000p86x33y27",
                "000p86x37y34",
            ),
        ),
        (
            TEST_POINTS[3],
            (-0.5, -0.5),
            0,
            0.001,
            ("000p86x32y60",),
        ),
        (
            TEST_POINTS[4],
            (0, 0),
            -62,
            0.2,
            (),
        ),
        (
            TEST_POINTS[5],
            (0, 0),
            188,
            0.25,
            (
                "135p90x69y52",
                "135p90x70y52",
                "135p90x69y51",
                "135p90x70y51",
                "135p90x68y52",
                "135p90x69y53",
                "135p90x68y51",
                "135p90x70y53",
                "135p90x71y52",
                "135p90x69y50",
                "135p90x71y51",
                "135p90x70y50",
                "135p90x68y53",
                "135p90x71y53",
                "135p90x68y50",
                "135p90x71y50",
            ),
        ),
    ],
)
def test_skycell_match(
    test_point, offset, rotation, size, expected_skycell_names, skymap_subset
):
    corners = mk_im_corners(*test_point + np.array(offset), rotation, size)

    intersecting_skycells = sm.find_skycell_matches(corners, skymap=skymap_subset)

    skycell_names = np.array(
        [skymap_subset.model.skycells[index]["name"] for index in intersecting_skycells]
    ).tolist()

    assert sorted(skycell_names) == sorted(expected_skycell_names)


@pytest.mark.parametrize(
    "test_point,expected_skycell_names",
    [
        (
            TEST_POINTS[1],
            [
                "000p86x69y62",
                "000p86x69y61",
                "000p86x68y62",
                "000p86x68y61",
                "000p86x69y63",
                "000p86x70y61",
                "000p86x67y62",
                "000p86x68y60",
            ],
        )
    ],
)
def test_match_from_wcs_with_bbox(test_point, expected_skycell_names, skymap_subset):
    wcsobj = mk_gwcs(
        *test_point,
        45,
        bounding_box=((-0.5, 4096 - 0.5), (-0.5, 4096 - 0.5)),
    )

    intersecting_skycells = sm.find_skycell_matches(wcsobj, skymap=skymap_subset)

    skycell_names = np.array(
        [skymap_subset.model.skycells[index]["name"] for index in intersecting_skycells]
    ).tolist()

    assert skycell_names == expected_skycell_names


@pytest.mark.parametrize("test_point", [TEST_POINTS[1]])
def test_match_from_wcs_without_bbox(test_point):
    wcsobj = mk_gwcs(*test_point, 45)

    with pytest.raises(ValueError):
        sm.find_skycell_matches(wcsobj, skymap=skymap_subset)
