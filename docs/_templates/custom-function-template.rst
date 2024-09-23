{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] %}
{% set parent_module = parts[:-1] | join('.') | escape %}
Back to :mod:`{{ parent_module }}`
{{ method_name | escape | underline }}
.. autosummary:: {{ fullname }}