"""Inject a step current at the soma."""

from biophysics_fitting.setup_stim import setup_soma_step


def soma_current_injection(cell, amplitude=None, delay=None, duration=None):
    """Inject a step current at the soma.
    
    Args:
        cell (:class:`~single_cell_parser.cell.Cell`): The cell object.
        amplitude (float): The amplitude of the current (nA).
        delay (float): The delay of the current (ms).
        duration (float): The duration of the current (ms).

    Returns:
        :class:`~single_cell_parser.cell.Cell`: The cell with the current injection set up.

    See also:
        :py:meth:`biophysics_fitting.setup_stim.setup_soma_step`
    """
    setup_soma_step(cell, amplitude=amplitude, delay=delay, duration=duration)
    return cell