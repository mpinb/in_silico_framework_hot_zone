{% set parent = fullname.split('.')[:-1] | join('.') | escape %}
{% if parent %}
Back to :mod:`{{ parent }}`
{% endif %}

{{ fullname.split('.')[-1] | escape | underline }}

.. This template decides how the class pages look
   autosummary directives decide what to include in the class page
   the for loops decide which entries to add to the autosummary directive


.. currentmodule:: {{ module }}

.. automethod:: {{ objname }}