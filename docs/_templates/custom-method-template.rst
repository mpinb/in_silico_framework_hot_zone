{% set parts = fullname.split('.') %}
{% set class_name = parts[-2] if parts|length > 2 else None %}
{% set method_name = parts[-1] %}


{% if class_name %}
{% set parent_module = parts[:-2] | join('.') | escape %}
Back to :py:class:`{{ parent_module }}.{{ class_name }}`
{{ class_name }}.{{ method_name | escape | underline }}
.. automethod:: {{ class_name }}.{{ method_name }}
{% else %}
{% set parent_module = parts[:-1] | join('.') | escape %}
Back to :mod:`{{ parent_module }}`
{{ method_name | escape | underline }}
.. autofunction:: {{ method_name }}
{% endif %}

.. currentmodule:: {{ parent_module }}