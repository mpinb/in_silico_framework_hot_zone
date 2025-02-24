.. graphviz::
    :caption: Sphinx and GraphViz Data Flow
    :alt: How Sphinx and GraphViz Render the Final Document
    :align: center

     digraph "sphinx-ext-graphviz" {
         size="6,4";
         rankdir="LR";
         graph [fontname="Verdana", fontsize="12"];
         node [fontname="Verdana", fontsize="12"];
         edge [fontname="Sans", fontsize="9"];

         morphology_file [label="morphology.hoc", shape="note", fontcolor=black,
                 fillcolor="white", style=filled,
                 xref=":ref:`hoc_file_format`"];
         scim [label="single_cell_input_mapper", shape="folder",
                 fontcolor=black, fillcolor="white", style=filled,
                 xref=":py:mod:`singlecell_input_mapper`"];
         biophys_df [label="biophys.df", shape="note", fontcolor=black,
                 fillcolor="white", style=filled];

         morphology_file -> scim [label="", fontcolor="black", fontsize="10"];
     }