"""
Vendored from https://github.com/dovahcrow/patchify.py (MIT License).

The upstream package declares numpy<2, which conflicts with tensorflow>=2.16.
Since patchify works fine with numpy 2.x (the constraint is just stale),
we vendor it here to avoid the dependency conflict.

view_as_windows is originally from scikit-image (BSD-3-Clause License).
See license text below.
"""

# =============================================================================
# view_as_windows — from scikit-image
# https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py
#
# Copyright (C) 2011, the scikit-image team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of skimage nor the names of its contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import numbers
from typing import Tuple, Union, cast

import numpy as np
from numpy.lib.stride_tricks import as_strided


Imsize = Union[Tuple[int, int], Tuple[int, int, int]]


# ---------------------------------------------------------------------------
# view_as_windows
# ---------------------------------------------------------------------------

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    being shifted by a single row or column (or an index of a higher
    dimension).

    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better known as hyperrectangle) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    """
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # Build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)
    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


# ---------------------------------------------------------------------------
# patchify / unpatchify — from dovahcrow/patchify.py (MIT License)
# ---------------------------------------------------------------------------

def patchify(image: np.ndarray, patch_size: Imsize, step: int = 1) -> np.ndarray:
    """
    Split a 2D or 3D image into small patches given the patch size.

    Parameters
    ----------
    image : ndarray
        The image to be split. It can be 2D (m, n) or 3D (m, n, c).
    patch_size : tuple
        The size of a single patch.
    step : int
        The step size between patches.

    Returns
    -------
    patches : ndarray
        Array of patches.

    Examples
    --------
    >>> image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> patches = patchify(image, (2, 2), step=1)
    >>> assert patches.shape == (2, 3, 2, 2)
    >>> reconstructed_image = unpatchify(patches, image.shape)
    >>> assert (reconstructed_image == image).all()
    """
    return view_as_windows(image, patch_size, step)


def unpatchify(patches: np.ndarray, imsize: Imsize) -> np.ndarray:
    """
    Merge patches back into the original image.

    Parameters
    ----------
    patches : ndarray
        The patches to merge. It can be patches for a 2D image (k, l, m, n)
        or 3D volume (i, j, k, l, m, n).
    imsize : tuple
        The size of the original image or volume.

    Returns
    -------
    image : ndarray
        Reconstructed image.

    Examples
    --------
    >>> image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> patches = patchify(image, (2, 2), step=1)
    >>> assert patches.shape == (2, 3, 2, 2)
    >>> reconstructed_image = unpatchify(patches, image.shape)
    >>> assert (reconstructed_image == image).all()
    """
    assert len(patches.shape) / 2 == len(
        imsize
    ), "The patches dimension is not equal to the original image size"

    if len(patches.shape) == 4:
        return _unpatchify2d(patches, cast(Tuple[int, int], imsize))
    elif len(patches.shape) == 6:
        return _unpatchify3d(patches, cast(Tuple[int, int, int], imsize))
    else:
        raise NotImplementedError(
            "Unpatchify only supports a matrix of 2D patches (k, l, m, n)"
            f" or 3D volumes (i, j, k, l, m, n), but got: {patches.shape}"
        )


def _unpatchify2d(
    patches: np.ndarray, imsize: Tuple[int, int]
) -> np.ndarray:
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)

    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    s_w = int(s_w)
    s_h = int(s_h)

    i, j = 0, 0

    while True:
        i_o, j_o = i * s_h, j * s_w
        image[i_o : i_o + p_h, j_o : j_o + p_w] = patches[i, j]

        if j < n_w - 1:
            j = min((j_o + p_w) // s_w, n_w - 1)
        elif i < n_h - 1 and j >= n_w - 1:
            i = min((i_o + p_h) // s_h, n_h - 1)
            j = 0
        elif i >= n_h - 1 and j >= n_w - 1:
            break
        else:
            raise RuntimeError("Unreachable")

    return image


def _unpatchify3d(
    patches: np.ndarray, imsize: Tuple[int, int, int]
) -> np.ndarray:
    assert len(patches.shape) == 6

    i_h, i_w, i_c = imsize
    image = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, n_c, p_h, p_w, p_c = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)
    s_c = 0 if n_c <= 1 else (i_c - p_c) / (n_c - 1)

    if int(s_w) != s_w:
        raise NonUniformStepSizeError(i_w, n_w, p_w, s_w)
    if int(s_h) != s_h:
        raise NonUniformStepSizeError(i_h, n_h, p_h, s_h)
    if int(s_c) != s_c:
        raise NonUniformStepSizeError(i_c, n_c, p_c, s_c)

    s_w = int(s_w)
    s_h = int(s_h)
    s_c = int(s_c)

    i, j, k = 0, 0, 0

    while True:
        i_o, j_o, k_o = i * s_h, j * s_w, k * s_c
        image[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] = patches[i, j, k]

        if k < n_c - 1:
            k = min((k_o + p_c) // s_c, n_c - 1)
        elif j < n_w - 1 and k >= n_c - 1:
            j = min((j_o + p_w) // s_w, n_w - 1)
            k = 0
        elif i < n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            i = min((i_o + p_h) // s_h, n_h - 1)
            j = 0
            k = 0
        elif i >= n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            break
        else:
            raise RuntimeError("Unreachable")

    return image


class NonUniformStepSizeError(RuntimeError):
    """Raised when unpatchify cannot reconstruct with a uniform step size."""

    def __init__(
        self, imsize: int, n_patches: int, patch_size: int, step_size: float
    ) -> None:
        super().__init__(imsize, n_patches, patch_size, step_size)
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.imsize = imsize
        self.step_size = step_size

    def __repr__(self) -> str:
        return (
            f"Unpatchify only supports reconstructing an image with a uniform step "
            f"size for all patches. However, reconstructing {self.n_patches} x "
            f"{self.patch_size}px patches to a {self.imsize}px image requires "
            f"{self.step_size} as step size, which is not an integer."
        )

    def __str__(self) -> str:
        return self.__repr__()
