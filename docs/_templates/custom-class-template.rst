{% set parent_module = fullname.split('.')[:-1] | join('.') %}
{% set class_name = fullname.split('.')[-1] %}
{% if parent_module %}
Back to :mod:`{{ parent_module | escape }}`
{% endif %}

{{ class_name | escape | underline }}

.. This template decides how the class pages look
   autosummary directives decide what to include in the class page
   the for loops decide which entries to add to the autosummary directive

.. currentmodule:: {{ parent_module  }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   
   {% set filtered_methods = methods | select("ne", "__init__") | list %}

   {% block methods %} 
   {% if filtered_methods %}

   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :template: custom-method-template.rst

      {% for item in filtered_methods %}
      {% if not item.meta or not item.meta.private %}
         {{ class_name }}.{{ item }}
      {% endif %}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::

   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}