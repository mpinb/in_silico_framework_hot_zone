.. setup:
{% import 'macros.rst' as macros %}
{% set parent_module = obj.id.split('.')[:-1] | join('.') | escape %}

.. backlink:

{% if parent_module %}
Back to :mod:`{{ parent_module }}`
{% endif %}

.. title:

{{ obj.id.split('.')[-1] | escape | underline }}

.. content:

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
