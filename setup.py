# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='macarico',
      version='0.0.1',
      description='',
      author=u'Hal Daumé III and Tim Vieira',
      packages=['macarico'],
      install_requires=[
          'tensorboardX', 'torch'  #'pytorch' <- should install via conda, not pip.
      ])
