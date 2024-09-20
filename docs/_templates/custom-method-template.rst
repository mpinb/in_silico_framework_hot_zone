{% set parts = fullname.split('.') %}
{% set parent_module = parts[:-2] | join('.') | escape %}
{% set class_name = parts[-2] if parts|length > 2 else None %}
{% set method_name = parts[-1] %}

.. currentmodule:: {{ parent_module }}

{% if class_name %}
Back to :py:class:`{{ parent_module }}.{{ class_name }}`
{% else %}
Back to :mod:`{{ parent_module }}`
{% endif %}

{% if class_name %}
{{ class_name }}.{{ method_name | escape | underline }}
.. automethod:: {{ class_name }}.{{ method_name }}
{% else %}
{{ method_name | escape | underline }}
.. autofunction:: {{ method_name }}
{% endif %}