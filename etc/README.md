# ETC
This directory simply provides a list of conda and pip packages for both he Python3 and Python2.7 environments for isf. These are used in the github workflows at .github/workflows to install the correct dependencies to run tests.

These are in a separate folder, since conda is being real annoying about from where and how it can read module lists. Moving these files to .github/ is a good way to break the environment for some reason. Probably due to the fact that .github starts with a leading dot.