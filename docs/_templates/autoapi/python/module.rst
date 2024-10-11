{% import 'macros.rst' as macros %}

{% set visible_children =
    module_object.children|selectattr("display")|rejectattr("imported")|list %}
{% set visible_classes =
    visible_children|selectattr("type", "equalto", "class")|list %}
{% set property_methods =
    all_methods|selectattr("properties", "contains", "property")|list %}

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