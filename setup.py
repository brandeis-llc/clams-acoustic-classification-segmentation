from setuptools import setup

with open('README.md') as readme:
    long_desc = readme.read()

with open('requirements.txt') as requirements:
    requires = requirements.readlines()

setup(
    name='brandeis-acs',
    version='1.9',
    packages=['bacs'],
    url='https://github.com/brandeis-llc/acoustic-classification-segmentation',
    license='MIT',
    author='Keigh Rim',
    author_email='krim@brandeis.edu',
    description='Brandeis Acoustic Classification & Segmentation tool',
    long_description=long_desc,
    long_description_content_type="text/markdown",
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
