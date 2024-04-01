import torch
import torch.fft as tfft
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

import time 
import inspect


class Results:
    r"""
    A minimal class for use when returning multiple values from a function

    This class supports dict-like assignment and retrieval
    (``obj['im'] = im``), namedtuple-like attribute look-ups (``obj.im``),
    and generic class-like object assignment (``obj.im = im``)

    """
    def __init__(self, **kwargs):
        self._func = inspect.getouterframes(inspect.currentframe())[1].function
        self._time = time.asctime()

    def __iter__(self):
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                yield v

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __str__(self):
        header = "―" * 78
        lines = [
            header,
            f"Results of {self._func} generated at {self._time}",
            header,
        ]
        for item in list(self.__dict__.keys()):
            if item.startswith('_'):
                continue
            if (isinstance(self[item], torch.Tensor)):
                s = tuple(self[item].size())
                lines.append("{0:<25s} Tensor of size {1}".format(item, s))
            elif isinstance(self[item], dict):
                N = len(self[item])
                lines.append("{0:<25s} Dictionary with {1} items".format(item, N))
            else:
                lines.append("{0:<25s} {1}".format(item, self[item]))
        lines.append(header)
        return "\n".join(lines)


def _get_radial_sum(dt, bins, bin_size, autocorr):
    radial_sum = torch.zeros(bins[:-1].shape, dtype=torch.float32).to(autocorr.device)
    
    for i, r in enumerate(bins[:-1]):
        mask = (dt <= r) * (dt > (r - bin_size[i]))
        mask = mask.to(torch.float32).to(autocorr.device)
        
        flattened_autocorr = autocorr.view(-1)
        flattened_mask = mask.view(-1)
        masked_values = flattened_autocorr * flattened_mask
        radial_sum[i] = torch.sum(masked_values) / torch.sum(mask)
    
    return radial_sum


def _parse_histogram(h, voxel_size=1, density=True):
    delta_x = h[1]
    P = h[0]
    bin_widths = delta_x[1:] - delta_x[:-1]
    temp = P * bin_widths
    C = torch.cumsum(temp.flip(0), dim=0).flip(0)
    S = P * bin_widths
    
    if not density:
        P /= torch.max(P)
        temp_sum = torch.sum(P * bin_widths)
        C /= temp_sum
        S /= temp_sum
    
    bin_edges = delta_x * voxel_size
    bin_widths = bin_widths * voxel_size
    bin_centers = ((delta_x[1:] + delta_x[:-1]) / 2) * voxel_size
    
    hist = Results()
    hist.pdf = P
    hist.cdf = C
    hist.relfreq = S
    hist.bin_centers = bin_centers
    hist.bin_edges = bin_edges
    hist.bin_widths = bin_widths
    
    return hist


def _radial_profile(autocorr, bins, pf=None, voxel_size=1):
    r"""
    Helper functions to calculate the radial profile of the autocorrelation

    Masks the image in radial segments from the center and averages the values
    The distance values are normalized and 100 bins are used as default.

    Parameters
    ----------
    autocorr : torch.Tensor
        The image of autocorrelation produced by FFT
    bins : torch.Tensor
        The edges of the bins to use in summing the radii, ** must be in voxels
    pf : float
        the phase fraction (porosity) of the image, used for scaling the
        normalized autocorrelation down to match the two-point correlation
        definition as given by Torquato
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : tpcf
    """
    if autocorr.dim() == 2:
        adj = torch.tensor(autocorr.shape, dtype=torch.float32).view(2, 1, 1)
        # use torch.round otherwise with odd image sizes, the mask generated can
        # be zero, resulting in Div/0 error
        inds = torch.stack(torch.meshgrid([torch.arange(d) for d in autocorr.shape])) - torch.round(adj / 2)
        dt = torch.sqrt(inds[0]**2 + inds[1]**2).to(autocorr.device)
    elif autocorr.dim() == 3:
        adj = torch.tensor(autocorr.shape, dtype=torch.float32).view(3, 1, 1, 1)
        # use torch.round otherwise with odd image sizes, the mask generated can
        # be zero, resulting in Div/0 error
        inds = torch.stack(torch.meshgrid([torch.arange(d) for d in autocorr.shape])) - torch.round(adj / 2)
        dt = torch.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2).to(autocorr.device)
    else:
        raise Exception('Image dimensions must be 2 or 3')
    
    if torch.max(bins) > torch.max(dt):
        msg = (
            'Bins specified distances exceeding maximum radial distance for'
            ' image size. Radial distance cannot exceed distance from center'
            ' of the image to the corner.'
        )
        raise Exception(msg)

    bin_size = bins[1:] - bins[:-1]
    radial_sum = _get_radial_sum(dt, bins, bin_size, autocorr)
    
    norm_autoc_radial = radial_sum / torch.max(autocorr)
    
    h = [norm_autoc_radial, bins]
    h = _parse_histogram(h, voxel_size=1)
    
    tpcf = Results()
    tpcf.distance = h.bin_centers * voxel_size
    tpcf.bin_centers = h.bin_centers * voxel_size
    tpcf.bin_edges = h.bin_edges * voxel_size
    tpcf.bin_widths = h.bin_widths * voxel_size
    tpcf.probability = norm_autoc_radial
    tpcf.probability_scaled = norm_autoc_radial * pf
    tpcf.pdf = h.pdf * pf
    tpcf.relfreq = h.relfreq
    
    return tpcf


def porosity(im):
    r"""
    Calculates the porosity of an image assuming 1's are void space and 0's
    are solid phase.

    All other values are ignored, so this can also return the relative
    fraction of a phase of interest in multiphase images.

    Parameters
    ----------
    im : torch.Tensor
        Image of the void space with 1's indicating void phase (or ``True``)
        and 0's indicating the solid phase (or ``False``). All other values
        are ignored (see Notes).

    Returns
    -------
    porosity : float
        Calculated as the sum of all 1's divided by the sum of all 1's and 0's.

    See Also
    --------
    phase_fraction
    find_outer_region

    Notes
    -----
    This function assumes void is represented by 1 and solid by 0, and all
    other values are ignored.  This is useful, for example, for images of
    cylindrical cores, where all voxels outside the core are labelled with 2.

    Alternatively, images can be processed with ``find_disconnected_voxels``
    to get an image of only blind pores.  This can then be added to the original
    image such that blind pores have a value of 2, thus allowing the
    calculation of accessible porosity, rather than overall porosity.

    --------
    """
    im = torch.as_tensor(im, dtype=torch.int64)
    Vp = torch.sum(im == 1, dtype=torch.int64)
    Vs = torch.sum(im == 0, dtype=torch.int64)
    e = Vp / (Vs + Vp).to(torch.float32)
    return e


def two_point_correlation(im, ddp: bool, rank: int, voxel_size=1, bins=100):
    r"""
    Calculate the two-point correlation function using Fourier transforms

    Parameters
    ----------
    im : torch.Tensor
        The image of the void space on which the 2-point correlation is
        desired, in which the phase of interest is labeled as True
    voxel_size : scalar
        The size of a voxel side in preferred units. The default is 1, so
        the user can apply the scaling to the returned results after the
        fact.
    bins : scalar or array_like
        Either an array of bin sizes to use, or the number of bins that
        should be automatically generated that span the data range. The
        maximum value of the bins, if passed as an array, cannot exceed
        the distance from the center of the image to the corner.

    Returns
    -------
    result : tpcf
        The two-point correlation function object, with named attributes:

        *distance*
            The distance between two points, equivalent to bin_centers
        *bin_centers*
            The center point of each bin. See distance
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``
        *probability_normalized*
            The probability that two points of the stated separation distance
            are within the same phase normalized to 1 at r = 0
        *probability* or *pdf*
            The probability that two points of the stated separation distance
            are within the same phase scaled to the phase fraction at r = 0

    Notes
    -----
    The Fourier transform approach utilizes the fact that the
    autocorrelation function is the inverse FT of the power spectrum
    density.

    Examples
    --------
    `Click here to view online example.`

    """
    if not ddp:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = rank
    # Move input image to GPU if it is not already
    if not im.is_cuda:
        im = im.to(device)

    # Get the phase fraction of the image
    pf = porosity(im) 
    
    if isinstance(bins, int):
        # Calculate half lengths of the image
        r_max = torch.tensor(im.shape[1]//2).to(torch.int)
        # r_max = (torch.ceil(torch.min(torch.tensor(im.shape))/2)).to(torch.int)
        
        # Get the bin size - ensures it will be at least 1
        bin_size = torch.ceil(r_max / bins).to(torch.int)
        
        # Calculate the bin divisions, equivalent to bin_edges
        bins = torch.arange(0, r_max + bin_size, bin_size).to(device)
    # Fourier Transform and shift image
    F = tfft.ifftshift(tfft.rfftn(tfft.fftshift(im)))
    import matplotlib.pyplot as plt
    # Compute Power Spectrum
    P = torch.abs(F)**2
    
    # Auto-correlation is the inverse of Power Spectrum
    autoc = torch.abs(tfft.ifftshift(tfft.irfftn(tfft.fftshift(P))))

    # Calculate the radial profile of the autocorrelation

    tpcf = _radial_profile(autoc, bins, pf=pf, voxel_size=voxel_size)
    
    return tpcf
