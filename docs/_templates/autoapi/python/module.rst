{% import 'macros.rst' as macros %}

{% set parent_module = obj.name.split('.')[:-1] | join('.') | escape %}
{% set shortname = obj.name.split('.')[-1] | escape %}

.. backlink:

{% if parent_module %}
Back to :mod:`{{ parent_module }}`
{% endif %}

.. title:

{{ shortname }}
{{ "=" * shortname|length }}

{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set property_methods = all_methods|selectattr("properties", "contains", "property")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "attribute")|list %}
{% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% set toctree_subs = visible_classes + visible_functions + visible_exceptions %}
{% set visible_subs = visible_subpackages + visible_submodules%}

.. a hidden toctree for sidebar navigation
.. Include all visible children, except for attributes (makes things verbose)
.. These names need to match either a python object in the Python space, or (as done here) a direct link to an .rst file (without the suffix), either relative or absolute.

.. toctree::
   :hidden:

   {% for sub in visible_subs %}
   {{ sub.name.split('.')[-1] }}/index
   {% endfor %}
   
   {% for child in toctree_subs %}
   {{ child.name.split('.')[-1] }}
   {% endfor %}

.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

      {% endif %}


{% if "show-module-summary" in autoapi_options %}
{% block classes scoped %}
{% if visible_classes %}
{% set public_classes = visible_classes|select('no_leading_underscore') %}
{% if public_classes %}

.. rubric:: Classes
{{ macros.auto_summary(public_classes, title="") }}
{% endif %}
{% endif %}
{% endblock %}

{% block functions scoped %}
{% if visible_functions %}
{% set public_functions = visible_functions|select('no_leading_underscore') %}
{% if public_functions %}

.. rubric:: Methods

{{ macros.auto_summary(public_functions, title="") }}
{% endif %}
{% endif %}
{% endblock %}

{% block subpackages %}
{% if visible_subs %}
{% set public_subs = visible_subs|select('no_leading_underscore') %}
{% if public_subs%}

.. rubric:: Modules
{{ macros.auto_summary(public_subs, title="") }}
{% endif %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if visible_exceptions %}

.. rubric:: Exceptions
{{ macros.auto_summary(visible_exceptions, title="") }}
{% endif %}
{% endblock %}

{% endif %}
