{% set parent_module = obj.id.split('.')[:-1] | join('.') | escape %}
{% set shortname = obj.name.split('.')[-1] | escape %}

.. backlink:

{% if parent_module %}
Back to :mod:`{{ parent_module }}`
{% endif %}


{% if obj.display %}
   {% if is_own_page %}

.. title:

{{ shortname }}
{{ "=" * shortname|length }}

   {% endif %}

.. py:function:: {{ obj.id }}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}
   {% for (args, return_annotation) in obj.overloads %}

                 {%+ if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
