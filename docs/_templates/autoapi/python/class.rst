
{% set visible_children = obj.children|selectattr("display")|list %}
{% set own_page_children = visible_children|selectattr("type", "in", own_page_types)|list %}
{% if is_own_page and own_page_children %}
{% if is_own_page and own_page_children %}
   {% set visible_attributes = own_page_children|selectattr("type", "equalto", "attribute")|list %}
   {% set visible_exceptions = own_page_children|selectattr("type", "equalto", "exception")|list %}
   {% set visible_classes = own_page_children|selectattr("type", "equalto", "class")|list %}
   {% set visible_methods = own_page_children|selectattr("type", "equalto", "method")|list %}
{% endif %}

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

