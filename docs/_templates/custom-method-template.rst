{% set parent = fullname.split('.')[:-2] | join('.') | escape %}
{% if parent %}
Back to :mod:`{{ parent }}`
{% endif %}

{{ fullname.split('.')[-1] | escape | underline }}

.. currentmodule:: {{ parent }}

.. automethod:: {{ objname }}