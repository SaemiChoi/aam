from setuptools import setup

setup(
    name='aam',
    version='0.0.1',
    description='beta',
    url='https://github.com/SaemiChoi/aam.git',
    author='SaemiChoi',
    license='saemi',
    packages=['aam'],
    zip_safe=False,
    install_requires=[
        'scikit-image',
        'diffusers==0.16.1',
        'transformers==4.27.4',
        'pandas',
        'nltk',
        'accelerate'
    ]

)