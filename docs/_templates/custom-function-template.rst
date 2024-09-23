{% set parts = fullname.split('.') %}
{% set method_name = parts[-1] %}
{% set parent_module = parts[:-1] | join('.') | escape %}
.. currentmodule:: {{ parent_module }}
Back to :py:mod:`{{ parent_module }}`
{{ method_name | escape | underline }}
.. autosummary:: {{ parent_module }}
    :members: {{ method_name }}