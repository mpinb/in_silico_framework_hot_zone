.. setup:
{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] %}
{% set class_name = parts[-2] %}
{% set parent_module = parts[:-2] | join('.') %}
{% set full_class_name = parent_module + '.' + class_name %}
{% set full_method_name = full_class_name + '.' + method_name %}

.. currentmodule:: {{ parent_module }}

.. backlink:
Back to :py:class:`{{ full_class_name }}`

.. title:
{{ class_name }}.{{ method_name | escape | underline }}

.. content:
.. note that this is a method (i.e. attribute of a class), not a function (i.e. attribute of a module)

.. automethod:: {{ full_method_name }}
