from setuptools import setup

with open('requirements.txt') as requirements:
    requires = requirements.readlines()

setup(
    name='brandeis-acs',
    version='1.6',
    packages=['bacs'],
    url='https://github.com/brandeis-llc/acoustic-classification-segmentation',
    license='MIT',
    author='Keigh Rim',
    author_email='krim@brandeis.edu',
    description='Brandeis Acoustic Classification & Segmentation tool',
    install_requires=requires,
    entry_points={
        'console_scripts': [
            'bacs = bacs.run:main'
        ]
    },
    package_data={
        'bacs': ['defmodel/*', 'defmodel/**/*']
    },

)
