#!/usr/bin/env python2

from distutils.core import setup

setup(
    name='hqp',
    description='Hierarchichal Quadratic Programming',
    version='1.0.1',
    packages=['hqp', 'hqp.multi_contact'],
    include_package_data=True,
    url='https://github.com/gepetto/hqp',
    author='Galo MALDONADO',
    author_email='galo.maldonado@laas.fr',
    license='LGPL',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
)
