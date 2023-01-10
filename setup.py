import os
from pathlib import Path
from setuptools import setup


def read(file_name):
    with open(
        os.path.join(
            Path(os.path.dirname(__file__)),
            file_name)
    ) as _file:
        return _file.read()


long_description = read('README.md')

setup(
    name='auto_annotate',
    version='1.0.5',
    description='Generate xml annotations for TensorFlow object detection models.',
    url='https://github.com/AlvaroCavalcante/auto_annotate',
    download_url='https://github.com/AlvaroCavalcante/auto_annotate',
    license='Apache License 2.0',
    author='Alvaro Leandro Cavalcante Carneiro',
    author_email='alvaroleandro250@gmail.com',

    py_modules=['auto_annotate', 'label_map_util', 'generate_xml'],
    packages=['protos'],
    package_dir={'': 'src'},
    
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[
        'tensorflow',
        'python',
        'python3',
        'object-detection',
        'annotation',
        'dataset',
        'semi-supervised',
        'deep-learning',
        'labeling'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    python_requires='>=3.8',
    install_requires=[
        'numpy==1.22.4',
        'tensorflow==2.11.0',
        'Pillow==9.3.0',
        'tqdm==4.64.1',
        'six==1.16.0'
    ]
)
