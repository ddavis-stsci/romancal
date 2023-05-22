from __future__ import annotations

import abc
from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roman_datamodels.datamodels import RampModel

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from astropy import units as u
from scipy import fft

# TODO:
# 2) test datamodel to standard view
# 3) create/test standard view to datamodel
# 4) create/test reference file to Coefficients


class Const(IntEnum):
    """
    The necessary assumed values for the reference pixel correction
        - REF: the number of columns used by the detector reference pixels
        - PAD: the number of columns to add to introduce the appropriate delay
               between channel reads from the guide window scan and other processes
               this enables uniform time sampling of the data.
        - CHAN_WIDTH: the number of columns in a channel (detector amplifier)
        - N_CHANNELS: the number of channels for the detector
        - N_COLUMNS: the total number of columns in the detector
    """

    REF = 4
    PAD = 12
    CHAN_WIDTH = 128
    N_DETECT_CHAN = 32
    N_COLUMNS = CHAN_WIDTH * N_DETECT_CHAN


def _extract_value(data):
    if isinstance(data, u.Quantity):
        if data.unit != u.DN:
            raise ValueError(f"Input data must be in units of DN, not {data.unit}")
        data = data.value

    return data


_offset = np.ndarray | None


@dataclass
class BaseView(abc.ABC):
    """Base description for the data views"""

    data: np.ndarray
    offset: _offset = None

    @abc.abstractproperty
    def detector(self) -> np.ndarray:
        """View of the detector pixels."""
        ...

    @abc.abstractproperty
    def left(self) -> np.ndarray:
        """View of the left reference pixels."""
        ...

    @abc.abstractproperty
    def right(self) -> np.ndarray:
        """View of the right reference pixels."""
        ...

    @abc.abstractproperty
    def amp33(self) -> np.ndarray:
        """View of the amp33 reference pixels."""
        ...


@dataclass
class StandardView(BaseView):
    """
    The "standard" view for the computation, as in it is the one which most closely
    represents the data orientations found in reality. This view places the detector
    and amp33 pixels in the same array, with the detector pixel columns on the left
    and amp33 pixel columns on the right (as opposed to in a separate array).

    The single common array enables us to vectorize the computation and minimize
    data copies.

    The data dimensions are [frame, row, column], where the columns are given by
        - [detector, amp33]
        - left = detector[:Const.REF]
        - right = detector[-Const.REF:]
    """

    @classmethod
    def from_datamodel(cls, datamodel: RampModel) -> StandardView:
        """
        Read the datamodel into the standard view.
        """
        detector = _extract_value(datamodel.data)

        # Extract amp33
        amp33 = _extract_value(datamodel.amp33)
        # amp33 is normally a uint16, but this computation requires it to match
        # the data type of the detector pixels.
        amp33 = amp33.astype(detector.dtype)

        return cls(np.dstack([detector, amp33]))

    @property
    def detector(self) -> np.ndarray:
        return self.data[:, :, : Const.N_COLUMNS]

    @property
    def left(self) -> np.ndarray:
        return self.detector[:, :, : Const.REF]

    @property
    def right(self) -> np.ndarray:
        return self.detector[:, :, -Const.REF :]

    @property
    def amp33(self) -> np.ndarray:
        return self.data[:, :, -Const.CHAN_WIDTH :]

    @property
    def channels(self) -> ChannelView:
        """
        The data split by channel and padded for uniform time sampling.
            The resulting view's dimensions are [frame, row, channel, column], where the
            columns are padded by Const.PAD columns of zeros.

        Note this is NOT a view of the data, but a copy.
        """
        frames, rows, columns = self.data.shape
        channels = columns // Const.CHAN_WIDTH

        # First split data into channels
        #    [frame, row, channel, channel_column]
        # Second reorder so channels are first (better memory alignment)
        #    [channel, frame, row, channel_column]
        data = self.data.reshape((frames, rows, channels, Const.CHAN_WIDTH)).transpose(
            (2, 0, 1, 3)
        )

        # Reverse every other channel column so columns are in reading order this is a
        # view of the data, not a copy at this point so this will back-propagate to the
        # data stored in the class.
        # This will need to be undone before returning the data because otherwise
        # the next call to this method will be different (columns will be erroneously
        # reversed a second time).
        # The next step of this method will pad the data, breaking the view, so we
        # can safely undo the changes afterwords.
        # We do not make an original copy of from the data so that we avoid having to
        # copy the data twice.
        # Maybe this should be done with a context manager?
        data[1::2, :, :, :] = data[1::2, :, :, ::-1]

        # pad channels with zeros to account for the pause at the end of each read
        # delay return until undoing the column reversal
        output = np.pad(
            data, ((0, 0), (0, 0), (0, 0), (0, Const.PAD)), constant_values=0
        )

        # Undo the column reversal
        data[1::2, :, :, :] = data[1::2, :, :, ::-1]

        return ChannelView(output, offset=self.offset)

    def remove_offset(self) -> StandardView:
        """
        Use linear least squares regression to remove the general linear offset
        from the data.
            - This records the offset in the class so that it can be returned to
              the data later.
            - Returns the class back to the user even though it will be modified
              in place by the views. This is so that it can be treated functionally.
        """

        frames, rows, columns = self.data.shape

        # Reshape data so that it is a view of the data of shape:
        #    [frame, frame_data]
        # where frame_data is the data for a single frame
        data = self.data.reshape((frames, rows * columns))

        # Craate an independent variable indexed by frame and centered at zero
        indep = np.arange(frames, dtype=data.dtype)
        indep = indep - np.mean(indep)

        # Compute sums needed for linear least squares
        sx = np.sum(indep)
        sxx = np.sum(indep**2)
        sy = np.sum(data, axis=0)
        sxy = np.matmul(indep, data, dtype=data.dtype)

        # Compute the offset (y-intercept) for the fit
        offset = (sy * sxx - sx * sxy) / (frames * sxx - sx**2)

        # Apply the offset to the data (in place)
        data -= offset

        self.offset = offset.reshape(rows, columns)

        return self

    def apply_offset(self) -> StandardView:
        """
        Apply the recorded offset in-place and return the class to the user.
            - If the offset has not been recorded, this will do nothing and return the
              object.
            - Returns the class back to the user even though it will be modified
              in place by the views. This is so that it can be treated functionally.
        """

        if self.offset is not None:
            self.data += self.offset
            self.offset = None

        return self


class ChannelView(BaseView):
    """
    This view is a transform of the standard view wherein the columns are split
    into a 4th dimension representing the each channel (amplifier) and the columns
    are then padded with zeros to account for the pause at the end of each read.

    The odd index channels are flipped so that the columns are indexed in the order
    in which the data is read.
    """

    @property
    def detector(self) -> np.ndarray:
        return self.data[: Const.N_DETECT_CHAN, :, :, :]

    @property
    def left(self) -> np.ndarray:
        """
        Left reference pixels, zero padded so that it has 140 columns to match the
        amp33 pixels.

        This is not a view, but a copy so that the data is not modified by the
        operation.
        """
        left = np.zeros(self.detector.shape[1:], dtype=self.detector.dtype)
        left[:, :, :4] = self.detector[0, :, :, :4]

        return left

    @property
    def right(self) -> np.ndarray:
        """
        Right reference pixels, zero padded so that it has 140 columns to match the
        amp33 pixels.

        This is not a view, but a copy so that the data is not modified by the
        operation.

        Note that due to channel flipping, the right reference pixels are the first
        four columns of the last detector channel. (backwards to initial guess)
        """
        right = np.zeros(self.detector.shape[1:], dtype=self.detector.dtype)
        right[:, :, :4] = self.detector[-1, :, :, :4]

        return right

    @property
    def amp33(self) -> np.ndarray:
        """
        amp33 reference pixels

        returns a view of the data, not a copy. This is to make some of the smoothing
        and interpolations occurr in place.
        """

        return self.data[-1, :, :, :]

    @property
    def standard(self) -> StandardView:
        """
        Output the standard view version of the data.
        """

        # Remove the padding from the data, copy is to break the view back to the
        # original data, this will be the only copy of the data needed.
        data = self.data[:, :, :, : -Const.PAD].copy()
        channels, frames, rows, columns = data.shape

        # Reverse every other channel column so columns are in standard order
        data[1::2, :, :, :] = data[1::2, :, :, ::-1]

        # Undo channel reordering to:
        #    [frame, row, channel, channel_column]
        # Second combine the channels into columns:
        #    [frame, row, column]
        data = data.transpose((1, 2, 0, 3)).reshape((frames, rows, columns * channels))

        return StandardView(data, offset=self.offset)

    def remove_trends(self) -> ChannelView:
        """
        Remove the (linear) trends from the frame boundary.
            - Change will be inplace of this object.
            - It will also return this object so it can be treated functionally.
        """
        channels, frames, rows, columns = self.data.shape

        # Create an independent variable indexed by frame and centered at zero
        t = np.arange(columns * rows, dtype=self.data.dtype).reshape((rows, columns))
        t = t - np.mean(t)

        # Locate the top and bottom reference pixel rows
        REF_ROWS = [*np.arange(Const.REF), *(rows - np.arange(Const.REF) - 1)[::-1]]

        # Restrict to the reference pixels and non-zero values
        t_ref = t[REF_ROWS, :]
        not_zero = self.data != 0

        # fit needs to be done for each channel and frame separately
        for chan_data, chan_not_zero in zip(self.data, not_zero):
            for frame_data, frame_not_zero in zip(chan_data, chan_not_zero):
                # Find the non-zero reference pixels for this channel and frame
                mask = frame_not_zero[REF_ROWS, :]

                # Compute the independent and dependent variables for the fit
                x_vals = t_ref[mask]
                y_vals = frame_data[REF_ROWS, :][mask]

                # Skip if there is no data
                if x_vals.size < 1 or y_vals.size < 1:
                    continue

                # Perform the fit using a 1st order polynomial, (i.e. linear fit)
                m, b = np.polyfit(x_vals, y_vals, 1)

                # Remove the fit from the data
                frame_data -= (t * m + b) * frame_not_zero

        return self

    def cosine_interpolate(self) -> ChannelView:
        """
        Perform Cosine weighted interpolation on the zero values of the amp33 columns.
        """
        data = self.amp33
        channels, _, rows, columns = self.detector.shape

        interp = np.sin(np.arange(1, channels + 1, dtype=data.dtype) * np.pi / channels)

        for frame in data:
            kern = frame.reshape(rows * columns)

            mask = (kern != 0).astype(np.int64)

            cov = np.convolve(kern, interp, mode="same")
            mask_conv = np.convolve(mask, interp, mode="same")

            kern = (cov / mask_conv).reshape(rows, columns)
            frame += kern * (1 - mask.reshape(rows, columns))

        # Fix NaN values
        np.nan_to_num(data, copy=False)

        return self

    @staticmethod
    def fft_interp_generator(
        frame: np.ndarray, pad: np.ndarray, apodize: np.ndarray
    ) -> np.ndarray:
        """
        Provide an infinite generator of interpolated frames using FFT interpolation,
        using the apodizing function provided.

            Since this is an iterative method, and determining the number of iterations
            is arbitrary, a generator is provided so tha it is simple to choose method
            of stopping the iteration (fixed, relative change, absolute change, etc.)
        """
        while True:
            result = apodize * fft.rfft(frame, workers=1) / frame.size
            result = fft.irfft(result * frame.size, workers=1).astype(frame.dtype)

            # Only update the padded columns
            frame[pad] = result[pad]

            yield frame

    def fft_interpolate(self, num: int = 3) -> ChannelView:
        """
        FFT interpolate the amp33 reference channel's added columns.

        Parameters:
            num: The number of iterations to perform. (default: 3)
        """
        frames, rows, columns = self.amp33.shape
        length = rows * columns

        # Mask all the data columns
        mask = np.ones((rows, columns), dtype=bool)
        mask[:, : -Const.PAD] = False
        mask = mask.flatten()

        # Find the indices of the padded columns
        pad = np.where(mask)[0]

        data = self.amp33.reshape(frames, length)
        apodize = (
            1 + np.cos(2 * np.pi * np.abs(np.fft.rfftfreq(length, 1 / length)) / length)
        ) / 2

        for frame in data:
            next(
                islice(  # advance the generator to the desired iteration
                    self.fft_interp_generator(frame, pad, apodize),
                    num - 1,  # next() advances and returns generator value
                    None,  # don't remember old iterations
                )
            )

        return self

    @staticmethod
    def forward_fft(channel: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Compute the forward FFT of the channel, and normalize if requested.
            - Normalize to just the non-padded columns, in this case 4 for the
              left and right reference pixels.
        """
        frames, rows, columns = channel.shape
        channel = channel.reshape(frames, rows * columns)

        channel = fft.rfft(channel / channel.shape[1])
        if normalize:
            channel *= columns / Const.REF

        return channel

    @property
    def reference_fft(self) -> ReferenceFFT:
        """
        Compute the reference FFTs object
        """
        return ReferenceFFT(
            self.forward_fft(self.left, True),
            self.forward_fft(self.right, True),
            self.forward_fft(self.amp33, False),
        )

    def correction(self, coeffs: Coefficients) -> np.ndarray:
        """
        Compute the correction array for the standard data
        """
        correction = self.reference_fft.correction(coeffs)

        return correction.reshape(self.data.shape).astype(self.data.dtype)

    def apply_correction(self, coeffs: Coefficients) -> StandardView:
        """
        Apply the correction and return the standard view of the data
        """
        self.data -= self.correction(coeffs)

        return self.standard


@dataclass
class ReferenceFFT:
    """
    FFTs of the Reference Pixels.

    Note that the FFTs have all been reshaped so that the rows and columns are combined
    into a single array. This means the FFTs are of shape
        - (frames, (rows * columns) // 2 + 1).

    where the // 2 + 1 is the length of the FFT for real data of even length
    """

    left: np.ndarray
    right: np.ndarray
    amp33: np.ndarray

    def channel_correction(self, coeffs: Coefficients) -> np.ndarray:
        """
        Generator, which yields the correction for each channel.
        """
        # We need to multiply by 2 because we are only using half of the FFT because
        # the data is real
        normalization = coeffs.gamma.shape[1] * 2

        for gamma, zeta, alpha in coeffs:
            correction = (
                np.multiply(self.left, gamma)
                + np.multiply(self.right, zeta)
                + np.multiply(self.amp33, alpha)
            ) * normalization

            # hold onto the previous correction so that shape is maintained
            # for the blank correction for the amp33 channel
            correction = fft.irfft(correction).real
            yield correction

        # Add zeros in for the amp33 channel as it does not get changed
        yield np.zeros(correction.shape)

    def correction(self, coeffs: Coefficients) -> np.ndarray:
        """
        Get the correction array for all of the data.
            - Stacks all channels into a single array

        Returns the correction array with the rows and columns in a single combined
        dimension, and the dtypes not adjusted to the original dtypes.
            - This will be handled by the ChannelView class
        """

        return np.array(list(self.channel_correction(coeffs)))


@dataclass
class Coefficients:
    gamma: np.ndarray
    zeta: np.ndarray
    alpha: np.ndarray

    def __iter__(self):
        return zip(self.gamma, self.zeta, self.alpha)
