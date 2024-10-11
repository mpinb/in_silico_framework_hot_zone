
{% import 'macros.rst' as macros %}
{% set parent_module = obj.id.split('.')[:-1] | join('.') %}
{% set class_name = obj.id.split('.')[-1] %}
{% if parent_module %}
Back to :mod:`{{ parent_module | escape }}`
{% endif %}

{{ class_name | escape | underline }}


{% if visible_methods or visible_attributes %}
.. rubric:: Overview

{% set summary_methods = visible_methods|rejectattr("properties", "contains", "property")|list %}
{% set summary_attributes = visible_attributes + visible_methods|selectattr("properties", "contains", "property")|list %}

{% if summary_attributes %}
{{ macros.auto_summary(summary_attributes, title="Attributes")|indent(3) }}
{% endif %}

{% if summary_methods %}
{{ macros.auto_summary(summary_methods, title="Methods")|indent(3) }}
{% endif %}

.. rubric:: Members

{% for attribute in visible_attributes %}
{{ attribute.render()|indent(3) }}
{% endfor %}
{% for method in visible_methods %}
{{ method.render()|indent(3) }}
{% endfor %}
{% endif %}
