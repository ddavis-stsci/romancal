"""Dark current step  code"""
# ! /usr/bin/env python

from romancal.dark_current import dark_current_subtract
from romancal.stpipe import RomanStep
from roman_datamodels import datamodels as rdm


__all__ = ["DarkCurrentStep"]


class DarkCurrentStep(RomanStep):
    """
    DarkCurrentStep: Performs dark current correction by subtracting
    dark current reference data from the input science data model.
    """

    spec = """
        dark_output = output_file(default = None) # Dark corrected model
    """

    reference_file_types = ['dark']

    def process(self, input):
        """ Code to remove the dark current signal from an observation."""
        # Open the input data model
        with rdm.open(input) as input_model:

            # Get the name of the dark reference file to use
            self.dark_name = self.get_reference_file(input_model, 'dark')
            self.log.info('Using DARK reference file %s', self.dark_name)

            # Check for a valid reference file
            if self.dark_name == 'N/A':
                self.log.warning('No DARK reference file found')
                self.log.warning('Dark current step will be skipped')
                result = input_model
                result.meta.cal_step.dark = 'SKIPPED'
                return result

            # Open the dark reference data file
            dark_ref_data = rdm.open(self.dark_name)

            # Check to make sure the science data and dark data have
            # the same shape
            if input_model.data.shape == dark_ref_data.data.shape:
                # Do the dark subtraction
                result = dark_current_subtract.dark_subtraction(
                    input_model, dark_ref_data)
            else:
                self.log.warning('DARK reference file and science data are \
                                 not the same shape')
                self.log.warning('Dark current step will be skipped')
                result.meta.cal_step.dark = 'SKIPPED'
                return result

        if self.save_results:
            try:
                self.suffix = 'darkcurrent'
            except AttributeError:
                self['suffix'] = 'darkcurrent'

        dark_ref_data.close()
        result.meta.cal_step.dark = 'COMPLETE'
        return result
