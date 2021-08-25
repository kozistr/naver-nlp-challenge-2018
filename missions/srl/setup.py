#nsml: nsml/default_ml:cuda9_torch1.0

from distutils.core import setup

setup(
    name='SRL_NSML_Baseline',
    version='1',
    description='SRL_NSML_Baseline',
    install_requires=[
        'tensorflow-gpu==2.5.1',
    ]
)
