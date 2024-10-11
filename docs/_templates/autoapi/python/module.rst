{% import 'macros.rst' as macros %}

{{ obj.name }}
{{ "=" * obj.name|length }}


{% set visible_children =
    obj.children|selectattr("display")|rejectattr("imported")|list %}
{% set visible_classes =
    visible_children|selectattr("type", "equalto", "class")|list %}
{% set property_methods =
    all_methods|selectattr("properties", "contains", "property")|list %}
{% set visible_functions =
      visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_attributes =
   visible_children|selectattr("type", "equalto", "attribute")|list %}
{% set visible_exceptions =
   visible_children|selectattr("type", "equalto", "exception")|list %}
{% set visible_submodules =
   visible_children|selectattr("type", "equalto", "module")|list %}



.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

      {% endif %}

      {% block subpackages %}
         {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
         {% if visible_subpackages %}
Subpackages
-----------

.. toctree::
   :maxdepth: 1

            {% for subpackage in visible_subpackages %}
   {{ subpackage.include_path }}
            {% endfor %}


         {% endif %}
      {% endblock %}
      {% block submodules %}
         {% set visible_submodules = obj.submodules|selectattr("display")|list %}
         {% if visible_submodules %}
Submodules
----------

.. toctree::
   :maxdepth: 1

            {% for submodule in visible_submodules %}
   {{ submodule.include_path }}
            {% endfor %}


         {% endif %}
      {% endblock %}
      {% block content %}
         {% set visible_children = obj.children|selectattr("display")|list %}
         {% if visible_children %}
            {% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
            {% if visible_attributes %}
               {% if "attribute" in own_page_types or "show-module-summary" in autoapi_options %}


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