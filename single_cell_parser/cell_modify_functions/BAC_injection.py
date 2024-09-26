'''Injects the BAC stimulus :cite:`Hay_Hill_Schuermann_Markram_Segev_2011` at a specified distance.'''
from biophysics_fitting.setup_stim import setup_BAC


def BAC_injection(cell, dist=None):
    '''Injects the BAC stimulus :cite:`Hay_Hill_Schuermann_Markram_Segev_2011` at a specified distance.

    Args:
        cell (:class:`~single_cell_parser.cell.Cell`): The cell object.
        dist (float): The distance from the soma (um).
    
    Returns:
        :class:`~single_cell_parser.cell.Cell`: The cell with the current injection set up.

    See also:
        :py:meth:`biophysics_fitting.setup_stim.setup_BAC`
    '''
    setup_BAC(cell, dist=dist)
    return cell