.. setup:
{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] if parts|length > 1 else 'undefined' %}
{% set class_name = parts[-2] if parts|length > 2 else 'undefined' %}
{% set parent_module = parts[:-2] | join('.') if parts|length > 2 else 'undefined' %}

.. note::
   parts: {{ parts }}
    method_name: {{ method_name }}
    class_name: {{ class_name }}
    parent_module: {{ parent_module }}
