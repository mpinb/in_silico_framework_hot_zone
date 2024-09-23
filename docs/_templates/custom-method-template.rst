{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] %}
{% set class_name = parts[-2] %}
{% set parent_module = parts[:-2] | join('.') | escape %}
.. currentmodule:: {{ parent_module }}
Back to :py:class:`{{ parent_module }}.{{ class_name }}`
{{ class_name }}.{{ method_name | escape | underline }}
.. automethod:: {{ class_name }}.{{ method_name }}