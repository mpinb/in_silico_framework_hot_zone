{% import 'macros.rst' as macros %}

{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% if visible_children %}
   {% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
   {% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
   {% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
   {% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
   {% set this_page_children = visible_children|rejectattr("type", "in", own_page_types)|list %}
{% endif %}
           
{% if "show-module-summary" in autoapi_options and (visible_classes or visible_functions) %}
{% block classes scoped %}
{% if visible_classes %}
{{ macros.auto_summary(visible_classes, title="Classes") }}
{% endif %}
{% endblock %}

{% block functions scoped %}
{% if visible_functions %}
{{ macros.auto_summary(visible_functions, title="Functions") }}
{% endif %}
{% endblock %}

{% block attributes scoped %}
{% if visible_attributes %}
{{ macros.auto_summary(visible_attributes, title="Attributes") }}
{% endif %}
{% endblock %}
{% endif %}