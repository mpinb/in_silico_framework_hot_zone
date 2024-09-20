{% set parent_module = fullname.split('.')[:-2] | join('.') | escape %}
{% set class_name = fullname.split('.')[-2] %}
{% set method_name = fullname.split('.')[-1] %}
{% if parent %}
Back to :mod:`{{ parent_module }}.{{ class_name }}`
{% endif %}

{{ fullname.split('.')[-1] | escape | underline }}


.. currentmodule:: {{ parent_module }}


.. automethod:: {{ class_name }}.{{ method_name }}