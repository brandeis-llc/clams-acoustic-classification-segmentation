from setuptools import setup

setup(
    name='brandeis-acs',
    version='1.2',
    packages=['bacs'],
    url='https://github.com/brandeis-llc/acoustic-classification-segmentation',
    license='MIT',
    author='Keigh Rim',
    author_email='krim@brandeis.edu',
    description='Brandeis Acoustic Classification & Segmentation tool',
    entry_points={
        'console_scripts': [
            'bacs = bacs.run:main'
        ]
    }
)
