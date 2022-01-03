import setuptools
from ae_sentence_embeddings.version import __version__

with open('README.md') as long_desc_file:
    long_description = long_desc_file.read()

setuptools.setup(
    name='ae_sentence_embeddings',
    version=__version__,
    author='nyekibence',
    author_email='nyeki.bence96@gmail.com',
    description='A package for ae_training Transformer-based autoencoders',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['tests', 'examples']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.9',
)
