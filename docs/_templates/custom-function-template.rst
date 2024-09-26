.. setup:

{% set parts = fullname.split('.') %}
{% set function_name = parts[-1] %}
{% set module_name = parts[-2] %}
{% set parent_module = parts[:-2] | join('.') %}
{% set full_module_name = parts[:-1] | join('.') %}
{% set full_function_name = full_module_name + '.' + function_name %}

.. currentmodule:: {{ full_module_name }}

.. backlink:

Back to :py:mod:`{{ full_module_name }}`

.. title:

{{ function_name | escape | underline }}

.. content:
.. note that this is a function (i.e. attribute of a module), not a method (i.e. attribute of a class)

.. autofunction:: {{ function_name }}
