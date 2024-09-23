.. setup:
{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] if parts|length > 1 else 'undefined' %}
{% set class_name = parts[-2] if parts|length > 2 else 'undefined' %}
{% set parent_module = parts[:-2] | join('.') if parts|length > 2 else 'undefined' %}
{% set full_class_name = (
    parent_module + '.' + class_name 
    if parent_module != 'undefined' and class_name != 'undefined' 
    else 'undefined'
) %}
{% set full_method_name = (
    full_class_name + '.' + method_name 
    if full_class_name != 'undefined' and method_name != 'undefined' 
    else 'undefined'
) %}

.. note::
   parts: {{ parts }}
    method_name: {{ method_name }}
    class_name: {{ class_name }}
    parent_module: {{ parent_module }}
    full_class_name: {{ full_class_name }}
    full_method_name: {{ full_method_name }}
