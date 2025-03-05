from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

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


def get_L_from_sec(sec, lv):
    out = 0
    for lv in range(1, lv + 1):
        assert lv - 1 >= 0
        out += get_dist(sec.pts[lv - 1], sec.pts[lv])
    return out


class Dendrogram:

    def __init__(self, cell, title="", colormap_synapses=None):
        # cell object needs to implement the following functionality:
        #     cell.soma
        #     section.children()
        #     section.L
        self.cell = cell
        self.title = title
        self.dendrogram_db = []
        self.dendrogram_db_by_name = {}
        if colormap_synapses:
            self.colormap_synapses = colormap_synapses
        else:
            self.colormap_synapses = defaultdict(lambda: "k")
        self._cell_to_dendrogram(self.cell.soma)
        self._soma_to_dendrogram(self.cell)
        self._set_initial_x()
        self._compute_main_bifurcation_section()

    def _add_synapse(self, l, label, x):
        if not "synapses" in l:
            l["synapses"] = {}
        if not label in l["synapses"]:
            l["synapses"][label] = []
        l["synapses"][label].append(x)

    def add_synapses(self, label, synlist, mode="from_anatomical_synapse_mapper"):
        """mode:

        'from_anatomical_synapse_mapper': to be used if synlist is generated from landmark as it is done
            with mikes manually mapped VPM synapses

        'id_relative_position': as in the syn file. Use something like [(syn.secID, syn.x) for syn in cell.synapses['VPM_C2']]
            to use synapses defined in a single_cell_parser.cell object
        """
        if mode == "from_anatomical_synapse_mapper":
            for syn in synlist:
                (sec_lv, seg_lv), (dist, x, point) = syn
                sec = self.cell.sections[sec_lv]
                l = self.get_db_by_sec(sec)
                x = get_L_from_sec(sec, seg_lv) + x + l["x_offset"] + l["x_dist_start"]
                self._add_synapse(l, label, x)
        elif mode == "id_absolute_position":
            for syn in synlist:
                sec_lv, x = syn
                sec = self.cell.sections[sec_lv]
                l = self.get_db_by_sec(sec)
                x = x + l["x_offset"] + l["x_dist_start"]
                self._add_synapse(l, label, x)
        elif mode == "id_relative_position":
            for syn in synlist:
                sec_lv, x = syn
                sec = self.cell.sections[sec_lv]
                l = self.get_db_by_sec(sec)
                x = x * sec.L + l["x_offset"] + l["x_dist_start"]
                self._add_synapse(l, label, x)

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
                {
                    "name": name,
                    "x_dist_start": x_dist_start,
                    "x_dist_end": x_dist_end,
                    "sec": sec,
                    "sec_id": self.cell.sections.index(sec),
                    "x_offset": x_offset,
                }
            )

            self._cell_to_dendrogram(sec, name, x_dist_end)
        self.dendrogram_db_by_name = {d_["name"]: d_ for d_ in self.dendrogram_db}

    def _soma_to_dendrogram(self, cell):
        soma_sections = [s for s in cell.sections if s.label.lower() == "soma"]
        print("adding {} soma sections".format(len(soma_sections)))
        for lv, sec in enumerate(soma_sections):
            self.dendrogram_db.append(
                {
                    "name": "Soma__{}__0".format(lv),
                    "x_dist_start": np.nan,
                    "x_dist_end": np.nan,
                    "sec": sec,
                    "x_offset": np.nan,
                }
            )

    def _set_initial_x(self):
        for lv, l in enumerate(
            sorted(self.dendrogram_db, key=lambda x: x["name"].split("__")[1])
        ):
            l["x"] = lv

    def plot(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(8, 10), dpi=200)
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(413)
        ax3 = fig.add_subplot(414)

        xlim = self._plot_dendrogram(ax)
        self._plot_dendrite_and_synapse_counts(ax2, xlim)
        self._plot_synapse_density(ax3, xlim)
        return fig

    def _plot_dendrogram(
        self, ax=None, colormap={"Dendrite": "grey", "ApicalDendrite": "r"}
    ):
        # uses: self.colormap_synapses
        if ax is None:
            fig = plt.figure(figsize=(8, 10), dpi=200)
            ax = fig.add_subplot(211)

        for lv, l in enumerate(self.dendrogram_db):
            label, tree, L = l["name"].split("__")
            if not label in ["Dendrite", "ApicalDendrite"]:
                continue
            x = l["x"]
            parent = self.get_parent_by_name(l["name"])
            if parent is None:
                parent_x = 0
            else:
                parent_x = parent["x"]
            ax.plot(
                [l["x_dist_start"], l["x_dist_start"] + l["x_offset"]],
                [x, x],
                ":",
                c="grey",
                linewidth=0.5,
            )
            ax.plot(
                [l["x_dist_start"] + +l["x_offset"], l["x_dist_end"]],
                [x, x],
                c=colormap[label],
                linewidth=0.5,
            )

            ax.plot(
                [l["x_dist_start"], l["x_dist_start"]],
                [x, parent_x],
                c=colormap[label],
                linewidth=0.5,
            )
            if "synapses" in l:
                for syntype in l["synapses"]:
                    for syn in l["synapses"][syntype]:
                        ax.plot(
                            [syn, syn],
                            [x + 0.5, x - 0.5],
                            c=self.colormap_synapses[syntype],
                            linewidth=0.3,
                        )

            if "main_bifurcation" in l:
                ax.plot([l["x_dist_end"]], [l["x"]], "o")
        ax.set_title(self.title)
        ax.set_ylabel("branch id")

        xlim = ax.get_xlim()
        return xlim

    def _compute_dendrite_and_synapse_counts(self, dist_end=None, binsize=50):
        if dist_end is None:
            dist_end = self._get_max_somadistance()
        bins = np.arange(0, dist_end + binsize, binsize)

        dendrite_density = [
            self.get_amount_of_dendrite_in_bin(bins[lv], bins[lv + 1])
            for lv in range(len(bins) - 1)
        ]
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
        dendrite_density_apical = [
            self.get_amount_of_dendrite_in_bin(
                bins[lv], bins[lv + 1], select=["ApicalDendrite"]
            )
            for lv in range(len(bins) - 1)
        ]
        dendrite_density_basal = [
            self.get_amount_of_dendrite_in_bin(
                bins[lv], bins[lv + 1], select=["Dendrite"]
            )
            for lv in range(len(bins) - 1)
        ]

        self.bins = bins
        self.dendrite_density = dendrite_density = np.array(dendrite_density)
        self.synapse_density = synapse_density = np.array(synapse_density)
        self.synapse_density_apical = synapse_density_apical = np.array(
            synapse_density_apical
        )
        self.synapse_density_basal = synapse_density_basal = np.array(
            synapse_density_basal
        )
        self.dendrite_density_apical = dendrite_density_apical = np.array(
            dendrite_density_apical
        )
        self.dendrite_density_basal = dendrite_density_basal = np.array(
            dendrite_density_basal
        )

    def _plot_dendrite_and_synapse_counts(self, ax, xlim, binsize=50):
        self._compute_dendrite_and_synapse_counts(xlim[1], binsize=binsize)
        histogram(
            (self.bins, self.dendrite_density),
            label="dendrite",
            colormap={"dendrite": "k"},
            fig=ax,
        )
        ax2 = ax.twinx()
        histogram(
            (self.bins, self.synapse_density),
            label="dendrite",
            colormap={"dendrite": "r"},
            fig=ax2,
        )
        ax.set_ylabel("dendritic length / bin", color="k")
        ax2.set_ylabel("# synapses / bin", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax.set_xlim(xlim)

    def _plot_synapse_density(self, ax, xlim, binsize=50):
        self._compute_dendrite_and_synapse_counts(xlim[1], binsize=binsize)
        histogram(
            (self.bins, self.synapse_density / self.dendrite_density),
            label="total",
            colormap={"total": "k"},
            fig=ax,
        )
        # I.histogram((bins, synapse_density_basal / dendrite_density_basal), label = 'basal',
        #             colormap = {'basal': 'grey'}, fig = ax)
        # I.histogram((bins, synapse_density_apical / dendrite_density_apical), label = 'apical',
        #             colormap = {'apical': 'red'}, fig = ax)
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylabel("# syn / micron dendritic length", color="k")
        ax.set_xlabel("somadistance / micron")

    def get_parent_by_name(self, name):
        sec = self.dendrogram_db_by_name[name]["sec"].parent
        if sec.label == "Soma":
            return None
        else:
            return self.get_db_by_sec(
                sec
            )  # next(l for l in self.dendrogram_db if l['sec'] == sec)

    def get_parent_dist_by_name(self, name):
        parent = self.get_parent_by_name(name)
        x = self.dendrogram_db_by_name[name]["x"]
        if parent is None:
            return x
        else:
            return x - parent["x"]

    def get_db_by_sec(self, sec):
        return next(l for l in self.dendrogram_db if l["sec"] == sec)

    def get_amount_of_dendrite_in_bin(
        self, min_, max_, select=["Dendrite", "ApicalDendrite"]
    ):

        out = 0
        for l in self.dendrogram_db:
            if not l["sec"].label in select:
                continue
            out += range_overlap(
                min_, max_, l["x_dist_start"], l["x_dist_end"], bool_=False
            )
        # if out < 50: return 0
        return out

    def get_number_of_synapses_in_bin(
        self, min_, max_, select=["Dendrite", "ApicalDendrite"], label=None
    ):
        out = 0
        for l in self.dendrogram_db:
            if not "synapses" in l:
                continue
            if not l["sec"].label in select:
                continue
            for label in list(l["synapses"].keys()):
                for syn in l["synapses"][label]:
                    if min_ <= syn < max_:
                        out += 1
        return out

    def _compute_main_bifurcation_section(self):
        try:
            sec = get_main_bifurcation_section(self.cell)
        except AssertionError:
            print("main bifurcation could not be identified!")
            self.main_bifur_dist = None
        else:
            l = self.get_db_by_sec(sec)
            l["main_bifurcation"] = True
            self.main_bifur_dist = l["x_dist_end"]

    def _get_max_somadistance(self):
        out = 0
        for l in self.dendrogram_db:
            out = max(out, l["x_dist_end"])
        self.max_somadist = out
        return out
