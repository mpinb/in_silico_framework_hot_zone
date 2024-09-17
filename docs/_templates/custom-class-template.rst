Back to :mod:`{{ fullname.split('.')[:-1] | join('.') | escape }}``

{{ fullname.split('.')[-1] | escape | underline }}

.. This template decides how the class pages look
   autosummary directives decide what to include in the class page
   the for loops decide which entries to add to the autosummary directive


.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :undoc-members:
   :special-members: __get__, __set__
   
   {% set filtered_methods = methods | select("ne", "__init__") | list %}

   {% block methods %} 
      {% if filtered_methods %}
      .. rubric:: {{ _('Methods') }}

      .. autosummary::
      {% for item in filtered_methods %}
         ~{{ name }}.{{ item }}
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