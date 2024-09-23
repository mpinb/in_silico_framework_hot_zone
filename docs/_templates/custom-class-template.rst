{% set parent_module = fullname.split('.')[:-1] | join('.') | escape %}
{% if parent_module %}
Back to :mod:`{{ parent_module }}`
{% endif %}

{{ fullname.split('.')[-1] | escape | underline }}

.. This template decides how the class pages look
   autosummary directives decide what to include in the class page
   the for loops decide which entries to add to the autosummary directive


.. autoclass:: {{ objname }}
   :show-inheritance:
   
   {% set filtered_methods = methods | select("ne", "__init__") | list %}

   {% block methods %} 
   {% if filtered_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:

      {% for item in filtered_methods %}
         {% if not item.meta or not item.meta.private %}
            {{ item.split('.')[-1] }}
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