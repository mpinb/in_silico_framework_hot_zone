{% import 'macros.rst' as macros %}

{{ obj.name }}
{{ "=" * obj.name|length }}

{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set property_methods = all_methods|selectattr("properties", "contains", "property")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "attribute")|list %}
{% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
{% set visible_submodules = visible_children|selectattr("type", "equalto", "module")|list %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}

.. a hidden toctree for the sidebar

.. toctree::
   :hidden:

   {% for child in visible_children %}
   :py:{{ child.type }}:{{ child.name }}
   {% endfor %}

.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

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

{% block subpackages %}
{% if visible_subpackages or visible_submodules %}

{{ macros.auto_summary(visible_subpackages + visible_submodules, title="Modules") }}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if visible_exceptions %}
{{ macros.auto_summary(visible_exceptions, title="Exceptions") }}
{% endif %}
{% endblock %}

{% endif %}
