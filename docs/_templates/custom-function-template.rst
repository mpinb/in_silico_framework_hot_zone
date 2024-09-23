.. setup:
{% set parts = fullname.split('.') %}
{% set function_name = parts[-1] %}
{% set class_name = parts[-2] %}
{% set parent_module = parts[:-2] | join('.') %}
{% set full_class_name = parent_module + '.' + class_name %}
{% set full_function_name = full_class_name + '.' + method_name %}
.. currentmodule:: {{ parent_module }}

.. backlink:
Back to :py:mod:`{{ parent_module }}`

.. title:
{{ function_name | escape | underline }}

.. content:
.. note that this is a function (i.e. attribute of a module), not a method (i.e. attribute of a class)
.. autofunction:: {{ function_name }}