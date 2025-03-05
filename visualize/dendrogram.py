import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from biophysics_fitting import get_main_bifurcation_section
from visualize.histogram import histogram


def range_overlap(beginning1, end1, beginning2, end2, bool_=True):
    out = min(end1, end2) - max(beginning1, beginning2)
    out = max(0, out)
    if bool_:
        out = bool(out)
    return out


def get_dist(x1, x2):
    assert len(x1) == len(x2)
    return np.sqrt(sum((xx1 - xx2) ** 2 for xx1, xx2 in zip(x1, x2)))


def _get_max_somadistance(dendrogram_db):
    out = 0
    for dendro_section in dendrogram_db:
        out = max(out, dendro_section.x_dist_end)
    return out


class _DendrogramSection:
    def __init__(
        self,
        name,
        x_dist_start,
        x_dist_end,
        sec,
        x_offset,
        main_bifurcation=False,
        sec_id=None,
    ):
        self.name = name
        self.x_dist_start = x_dist_start
        self.x_dist_end = x_dist_end
        self.sec = sec
        self.x_offset = x_offset
        self.synapses = {}
        self.main_bifurcation = main_bifurcation
        self.sec_id = sec_id


class Dendrogram:
    """Plot a dendrogram, distance and synapse statistics.

    Dendrograms are schematic representations of neuron morphologies.
    """

    def __init__(self, cell, title=""):
        self.cell = cell
        self.title = title
        self.dendrogram_db = []
        self._cell_to_dendrogram(basesec=self.cell.soma)
        self._soma_to_dendrogram(self.cell)
        self.dendrogram_db_by_name = {d_.name: d_ for d_ in self.dendrogram_db}
        self.dendrogram_db_by_sec_id = {int(d_.sec_id): d_ for d_ in self.dendrogram_db}
        self._compute_main_bifurcation_section()

    def _get_db_by_sec(self, sec):
        return next(
            dendro_section
            for dendro_section in self.dendrogram_db
            if dendro_section.sec == sec
        )

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        ax = self._plot_dendrogram(ax)
        return ax

    def get_parent_by_name(self, name):
        sec = self.dendrogram_db_by_name[name].sec.parent
        if sec.label == "Soma":
            return None
        else:
            return self._get_db_by_sec(sec)

    def _compute_main_bifurcation_section(self):
        try:
            sec = get_main_bifurcation_section(self.cell)
        except AssertionError:
            print("main bifurcation could not be identified!")
            self.main_bifur_dist = None
        else:
            dendro_section = self._get_db_by_sec(sec)
            dendro_section.main_bifurcation = True
            self.main_bifur_dist = dendro_section.x_dist_end

    def _cell_to_dendrogram(self, basesec=None, basename="", x_dist_base=0):
        if basename:
            basename = basename.split("__")
        else:
            basename = ("", "", "")
        for lv, sec in enumerate(sorted(basesec.children(), key=lambda x: x.label)):
            if sec.label not in ("Dendrite", "ApicalDendrite", "Soma"):
                continue

            x_offset = get_dist(basesec.pts[-1], sec.pts[0])
            x_dist_start = x_dist_base
            x_dist_end = x_dist_start + x_offset + sec.L

            name = (
                sec.label
                + "__"
                + (basename[1] + "_" if basename[1] else "")
                + str(lv)
                + "__"
                + str(np.round(sec.L))
            )
            self.dendrogram_db.append(
                _DendrogramSection(
                    name=name,
                    x_dist_start=x_dist_start,
                    x_dist_end=x_dist_end,
                    sec=sec,
                    x_offset=x_offset,
                    sec_id=sec.secID,
                )
            )

            self._cell_to_dendrogram(basesec=sec, basename=name, x_dist_base=x_dist_end)

    def _soma_to_dendrogram(self, cell):
        soma_sections = [s for s in cell.sections if s.label.lower() == "soma"]
        for lv, sec in enumerate(soma_sections):
            self.dendrogram_db.append(
                _DendrogramSection(
                    name="Soma__{}__0".format(lv),
                    x_dist_start=np.nan,
                    x_dist_end=np.nan,
                    sec=sec,
                    x_offset=np.nan,
                    sec_id=sec.secID,
                )
            )

    def _plot_dendrogram(
        self, ax, colormap={"Dendrite": "grey", "ApicalDendrite": "r"}
    ):
        lines = []

        for sec_id, dendro_section in enumerate(self.dendrogram_db):
            label, tree, L = dendro_section.name.split("__")
            if label not in ["Dendrite", "ApicalDendrite"]:
                continue
            sec_id = dendro_section.sec_id
            parent = self.get_parent_by_name(dendro_section.name)
            parent_sec_id = parent.sec_id if parent is not None else 0

            # Collect lines for batch plotting
            lines.append(
                (
                    [dendro_section.x_dist_start, parent_sec_id],
                    [dendro_section.x_dist_start, sec_id],
                    [dendro_section.x_dist_end, sec_id],
                )
            )

            if dendro_section.main_bifurcation:
                ax.plot([dendro_section.x_dist_end], [dendro_section.sec_id], "o")

        # Plot all collected lines at once
        line_segments = LineCollection(
            lines, colors=[colormap.get(label, "k")] * len(lines), linewidths=0.5
        )
        ax.add_collection(line_segments)

        ax.set_title(self.title)
        ax.set_ylabel("branch id")
        max_x = _get_max_somadistance(self.dendrogram_db)
        ax.set_xlim(-0.01 * max_x, max_x)
        ax.set_ylim(-5, len(self.dendrogram_db))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        return ax


class _DendrogramDendriteStatistics:
    def __init__(self, dendrogram_db, colormap_synapses=None):
        if colormap_synapses is None:
            colormap_synapses = {}
        self.dendrogram_db = dendrogram_db
        self.colormap_synapses = colormap_synapses

    def plot(self, ax=None):
        ax = self._plot_dendrogram(ax)
        xlim = ax.get_xlim()
        self._plot_dendrite_hist(ax, xlim)
        return ax

    def _get_amount_of_dendrite_in_bin(
        self, min_, max_, select=["Dendrite", "ApicalDendrite"]
    ):
        out = 0
        for dendro_section in self.dendrogram_db:
            if not dendro_section.sec.label in select:
                continue
            out += range_overlap(
                min_,
                max_,
                dendro_section.x_dist_start,
                dendro_section.x_dist_end,
                bool_=False,
            )
        return out

    def _plot_dendrite_hist(self, ax, xlim, binsize=50):
        self._compute_dendrite_hist(xlim[1], binsize=binsize)
        histogram(
            (self.bins, self.dendrite_density),
            label="dendrite",
            colormap={"dendrite": "k"},
            ax=ax,
        )
        ax.set_ylabel("dendritic length / bin", color="k")
        ax.set_xlim(xlim)
        return ax

    def _compute_dendrite_hist(self, dist_end=None, binsize=50):
        if dist_end is None:
            dist_end = _get_max_somadistance(self.dendrogram_db)
        bins = np.arange(0, dist_end + binsize, binsize)

        dendrite_density = [
            self._get_amount_of_dendrite_in_bin(bins[lv], bins[lv + 1])
            for lv in range(len(bins) - 1)
        ]

        dendrite_density_apical = [
            self._get_amount_of_dendrite_in_bin(
                bins[lv], bins[lv + 1], select=["ApicalDendrite"]
            )
            for lv in range(len(bins) - 1)
        ]
        dendrite_density_basal = [
            self._get_amount_of_dendrite_in_bin(
                bins[lv], bins[lv + 1], select=["Dendrite"]
            )
            for lv in range(len(bins) - 1)
        ]

        self.bins = bins
        self.dendrite_density = dendrite_density = np.array(dendrite_density)
        self.dendrite_density_apical = dendrite_density_apical = np.array(
            dendrite_density_apical
        )
        self.dendrite_density_basal = dendrite_density_basal = np.array(
            dendrite_density_basal
        )


class _DendrogramSynapseStatistics:
    def __init__(self, dendrogram_db, cell, colormap_synapses=None):
        if colormap_synapses is None:
            colormap_synapses = {}
        self.dendrogram_db = dendrogram_db
        self.cell = cell
        self.bins = None
        self.synapse_density = None
        self.dendrite_density = None
        self._add_synapses()
        self._compute_synapse_statistics()
        self._compute_synapse_hist()

    def _get_db_by_sec(self, sec):
        return next(
            dendro_section
            for dendro_section in self.dendrogram_db
            if dendro_section.sec == sec
        )

    def _add_synapse(self, dendro_section, label, x):
        if not label in dendro_section.synapses:
            dendro_section.synapses[label] = []
        dendro_section.synapses[label].append(x)

    def _add_synapses(self):
        """ """
        for cell_type, synapses in self.cell.synapses.items():
            for syn in synapses:
                sec_id = syn.secID
                sec = self.cell.sections[sec_id]
                dendro_sec = self._get_db_by_sec(sec)
                x = sec.L * syn.x + dendro_sec.x_dist_start
                self._add_synapse(dendro_sec, label=cell_type, x=x)

    def get_number_of_synapses_in_bin(
        self, min_, max_, select=["Dendrite", "ApicalDendrite"], label=None
    ):
        out = 0
        for dendro_section in self.dendrogram_db:
            if not dendro_section.sec.label in select:
                continue
            for label in list(dendro_section.synapses.keys()):
                for syn in dendro_section.synapses[label]:
                    if min_ <= syn < max_:
                        out += 1
        return out

    def _compute_synapse_hist(self, dist_end=None, binsize=50):
        if dist_end is None:
            dist_end = _get_max_somadistance(self.dendrogram_db)
        bins = np.arange(0, dist_end + binsize, binsize)

        synapse_density = [
            self.get_number_of_synapses_in_bin(bins[lv], bins[lv + 1])
            for lv in range(len(bins) - 1)
        ]
        synapse_density_apical = [
            self.get_number_of_synapses_in_bin(
                bins[lv], bins[lv + 1], select=["ApicalDendrite"]
            )
            for lv in range(len(bins) - 1)
        ]
        synapse_density_basal = [
            self.get_number_of_synapses_in_bin(
                bins[lv], bins[lv + 1], select=["Dendrite"]
            )
            for lv in range(len(bins) - 1)
        ]

        self.bins = bins
        self.synapse_density = synapse_density = np.array(synapse_density)
        self.synapse_density_apical = synapse_density_apical = np.array(
            synapse_density_apical
        )
        self.synapse_density_basal = synapse_density_basal = np.array(
            synapse_density_basal
        )

    def _compute_synapse_statistics(self):
        self.synapse_statistics = {}
        for dendro_section in self.dendrogram_db:
            for label in dendro_section.synapses:
                if label not in self.synapse_statistics:
                    self.synapse_statistics[label] = []
                self.synapse_statistics[label].extend(dendro_section.synapses[label])

    def _plot_synapse_density_hist(self, ax, xlim, binsize=50):
        self._compute_synapse_hist(xlim[1], binsize=binsize)
        if self.colormap_synapses is None:
            # Plot all
            self.colormap_synapses = {"total": "r"}
            histogram(
                (self.bins, self.synapse_density),
                label="total",
                colormap={"total": "r"},
                ax=ax,
            )
        else:
            # plot per label
            for label in self.colormap_synapses:
                histogram(
                    (self.bins, self.synapse_statistics[label]),
                    label=label,
                    colormap=self.colormap_synapses,
                    ax=ax,
                )
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylabel("# syn / micron dendritic length", color="k")

    def _plot_synapse_hist(self, ax, dendrite_density):
        histogram(
            (self.bins, self.synapse_density / dendrite_density),
            label="total",
            colormap={"total": "k"},
            ax=ax,
        )
        ax.legend()
        ax.set_ylabel("# syn / micron dendritic length", color="k")

    def _plot_synapses_dendrogram_overlay(self, ax):
        syn_lines = []
        syn_colors = []
        for dendro_section in self.dendrogram_db:
            sec_id = dendro_section.sec_id
            for syntype in dendro_section.synapses:
                for syn in dendro_section.synapses[syntype]:
                    syn_lines.append(
                        (
                            [syn, sec_id + 0.5],
                            [syn, sec_id - 0.5],
                        )
                    )
                    syn_colors.append(self.colormap_synapses.get(syntype, "k"))
        syn_line_segments = LineCollection(syn_lines, colors=syn_colors, linewidths=0.3)
        ax.add_collection(syn_line_segments)
        return ax

    def plot(self, ax=None, colormap_synapses=None):
        if ax is None:
            fig = plt.figure(figsize=(8, 10), dpi=200)
            ax = fig.add_subplot(211)

        ax = self._plot_synapse_hist(ax, colormap_synapses=colormap_synapses)
        return fig


class DendrogramStatistics(Dendrogram):
    def __init__(self, cell):
        self.dendrogram = Dendrogram(cell)
        self.dend_statistics = _DendrogramDendriteStatistics(
            self.dendrogram.dendrogram_db
        )
        self.syn_statistics = _DendrogramSynapseStatistics(
            self.dendrogram.dendrogram_db, cell=cell
        )

    def plot(self, figsize=None, colormap_synapses=None):
        self.syn_statistics.colormap_synapses = colormap_synapses
        if figsize is None:
            figsize = (8, 10)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        ax = self.dendrogram._plot_dendrogram(ax)
        xlim = ax.get_xlim()
        ax2 = self.dend_statistics._plot_dendrite_hist(ax2, xlim=xlim)
        ax2_2 = ax2.twinx()
        ax2_2 = self.syn_statistics._plot_synapse_density_hist(ax2_2, xlim=xlim)
        ax3 = self.syn_statistics._plot_synapse_hist(
            ax3, dendrite_density=self.dend_statistics.dendrite_density
        )
        ax = self.syn_statistics._plot_synapses_dendrogram_overlay(ax)
        return fig