#!/bin/bash
# Requirements installation for conll2015
#
# Author: GW [http://gw.tnode.com/] <gw.2015@tnode.com>
# License: All rights reserved

NAME="(`basename $(pwd)`)"
SRC="venv/src"
SITE_PACKAGES='venv/lib/python*/site-packages'
DIST_PACKAGES='/usr/lib/python*/dist-packages'

virtualenv --prompt="$NAME" venv
source venv/bin/activate
[ ! -e "$SRC" ] && mkdir "$SRC"

# Prerequisites for theano
sudo aptitude install python-numpy=1:1.8.2-2 python-scipy=0.14.0-2
[ -d $DIST_PACKAGES/numpy ] && cp -a $DIST_PACKAGES/numpy* $SITE_PACKAGES
[ -d $DIST_PACKAGES/scipy ] && cp -a $DIST_PACKAGES/scipy* $SITE_PACKAGES

# Prerequisites for matplotlib
sudo aptitude install libfreetype6-dev

# Requirements
pip install joblib
pip install gensim
pip install theano
pip install tabulate
pip install matplotlib

echo
echo "Use: . venv/bin/activate"
echo
