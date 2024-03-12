"""Suffix manipulation

API
---
`remove_suffix`
    Main function used to remove suffixes from
    file base names.

Notes
-----

`KNOW_SUFFIXES` is the list used by `remove_suffix`. This
list is generated by the function `combine_suffixes`. The function uses
`SUFFIXES_TO_ADD` to add other suffixes that it would otherwise not
discover or should absolutely be in the list. The function uses
'SUFFIXES_TO_DISCARD` for strings found that are not to be considered
suffixes.

Hence, to update `KNOW_SUFFIXES`, update both `SUFFIXES_TO_ADD` and
`SUFFIXES_TO_DISCARD` as necessary, then use the output of
`find_suffixes`.
"""

import itertools
import logging
import re

__all__ = ["remove_suffix"]

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Suffixes that are hard-coded or otherwise
# have to exist. Used by `find_suffixes` to
# add to the result it produces.
SUFFIXES_TO_ADD = [
    "ca",
    "crf",
    "cal",
    "dark",
    "flat",
    "median",
    "phot",
    "photom",
    "ramp",
    "rate",
    "uncal",
    "assignwcs",
    "dq_init",
    "dqinit",
    "assign_wcs",
    "linearity",
    "jump",
    "rampfit",
    "saturation",
    "sourcedetection",
    "dark_current",
    "darkcurrent",
    "outlier_detection",
    "skymatch",
    "refpix",
]

# Suffixes that are discovered but should not be considered.
# Used by `find_suffixes` to remove undesired values it has found.
SUFFIXES_TO_DISCARD = ["highlevelpipeline", "pipeline", "step"]


# Calculated suffixes.
_calculated_suffixes = {
    "saturationstep",
    "darkcurrentstep",
    "jumpstep",
    "rampfit",
    "dark_current",
    "assignwcsstep",
    "flatfieldstep",
    "fluxstep",
    "step",
    "dqinitstep",
    "assign_wcs",
    "linearity",
    "rampfitstep",
    "photomstep",
    "pipeline",
    "dq_init",
    "linearitystep",
    "dark_current",
    "jump",
    "sourcedetection",
    "sourcedetectionstep",
    "tweakregstep",
    "outlierdetectionstep",
    "skymatchstep",
    "refpixstep",
    "resamplestep",
}


# ##########
# Suffix API
# ##########
def remove_suffix(name):
    """Remove the suffix if a known suffix is already in name"""
    separator = None
    match = REMOVE_SUFFIX_REGEX.match(name)
    try:
        name = match.group("root")
        separator = match.group("separator")
    except AttributeError:
        pass
    if separator is None:
        separator = "_"
    return name, separator


def replace_suffix(name, new_suffix):
    """Replace suffix on name

    Parameters
    ----------
    name: str
        The name to replace the suffix of.
        Expected to be only the basename; no extensions.

    new_suffix:
        The new suffix to use.
    """
    no_suffix, separator = remove_suffix(name)
    return no_suffix + separator + new_suffix


# #####################################
# Functions to generate `KNOW_SUFFIXES`
# #####################################
def combine_suffixes(
    to_add=(_calculated_suffixes, SUFFIXES_TO_ADD), to_remove=(SUFFIXES_TO_DISCARD,)
):
    """Combine the suffix lists into a single list

    Parameters
    ----------
    to_add: [iterable[, ...]]
        List of iterables to add to the combined list.

    to_remove: [iterable[, ...]]
        List of iterables to remove from the combined list.

    Returns
    -------
    suffixes: list
        The list of suffixes.
    """
    combined = set(itertools.chain.from_iterable(to_add))
    combined.difference_update(itertools.chain.from_iterable(to_remove))
    combined = list(combined)
    combined.sort()

    return combined


def find_suffixes():
    """Find all possible suffixes from the romancal package

    Returns
    -------
    suffixes: set
        The set of all programmatically findable suffixes.

    Notes
    -----
    This will load all of the `romancal` package. Consider if this
    is worth doing dynamically or only as a utility to update
    a static list.
    """
    from romancal.stpipe.utilities import all_steps

    # First traverse the code base and find all
    # `Step` classes. The default suffix is the
    # class name.
    suffixes = {klass_name.lower() for klass_name, klass in all_steps().items()}

    # That's all folks
    return list(suffixes)


# --------------------------------------------------
# The set of suffixes used by the pipeline.
# This set is generated by `combine_suffixes`.
# Only update this list by `combine_suffixes`.
# Modify `SUFFIXES_TO_ADD` and `SUFFIXES_TO_DISCARD`
# to change the results.
# --------------------------------------------------
KNOW_SUFFIXES = combine_suffixes()

# Regex for removal
REMOVE_SUFFIX_REGEX = re.compile(
    "^(?P<root>.+?)((?P<separator>_|-)(" + "|".join(KNOW_SUFFIXES) + "))?$"
)


# ############################################
# Main
# Find and report differences from known list.
# ############################################
if __name__ == "__main__":
    print("Searching code base for calibration suffixes...")
    calculated_suffixes = find_suffixes()
    found_suffixes = combine_suffixes(
        to_add=(calculated_suffixes, SUFFIXES_TO_ADD), to_remove=(SUFFIXES_TO_DISCARD,)
    )
    print(
        "Known list has {known_len} suffixes. Found {new_len} suffixes.".format(
            known_len=len(KNOW_SUFFIXES), new_len=len(found_suffixes)
        )
    )
    print(
        "Suffixes that have changed are"
        f" {set(found_suffixes).symmetric_difference(KNOW_SUFFIXES)}"
    )
