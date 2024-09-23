{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] %}
{% set class_name = parts[-2] if parts|length > 2 else None %}

{% if class_name %}
   {% set parent_module = parts[:-2] | join('.') | escape %}
   .. currentmodule:: {{ parent_module }}
   Back to :py:class:`{{ parent_module }}.{{ class_name }}`
   {{ class_name }}.{{ method_name | escape | underline }}
{% else %}
   {% set parent_module = parts[:-1] | join('.') | escape %}
   .. currentmodule:: {{ parent_module }}
   Back to :mod:`{{ parent_module }}`
   {{ method_name | escape | underline }}
{% endif %}

.. autosummary::
   {{ method_name }}