.. setup:
{% set parent_module = fullname.split('.')[:-1] | join('.') | escape %}

.. backlink:
{% if parent_module %}
Back to :mod:`{{ parent_module }}`
{% endif %}

.. title:
{{ fullname.split('.')[-1] | escape | underline }}

.. content:
.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   
   .. rubric:: Module attributes

   .. autosummary::
   
      {% for item in attributes %}
         {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}

   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
      :template: custom-function-template.rst

      {% for item in functions %}
      {% if not item.meta or not item.meta.private %}
         {{ item }}
      {% endif %}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :template: custom-class-template.rst
      
      {% for item in classes %}
         {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:

      {% for item in exceptions %}
         {{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}

.. rubric:: {{ _('Modules') }}

.. autosummary::
   :toctree:
   :template: custom-module-template.rst

   {% for item in modules %}
      {{ item.split('.')[-1] }}
   {%- endfor %}
{% endif %}
{% endblock %}
