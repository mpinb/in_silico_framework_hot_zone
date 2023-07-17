#!/abast/anaconda3_isf/bin/python

# -*- coding: utf-8 -*-
"""
This is the test that runs on CI on github
It can run both on the github servers and discover with pytest, as well as locally. You can do it with the same commands.
"""
import re
import sys
import distributed
import six

from py.test import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
