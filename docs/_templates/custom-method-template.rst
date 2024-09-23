.. setup:
{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] %}
{% set class_name = parts[-2] %}
{% set parent_module = parts[:-2] | join('.') %}
{% set full_class_name = parent_module + '.' + class_name %}
{% set full_method_name = full_class_name + '.' + method_name %}

{% if method_name %}
    .. currentmodule:: {{ parent_module }}

    Back to :py:class:`{{ full_class_name }}`

    {{ class_name }}.{{ method_name | escape | underline }}

    .. automethod:: {{ full_method_name }}
{% endif %}