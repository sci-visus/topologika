import platform
import setuptools

# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class GetNumpyInclude(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

module = setuptools.Extension(
    'topologika_reference',
    sources=['topologikareferencemodule.c'],
    include_dirs=[GetNumpyInclude()]
)

setuptools.setup(
    name='topologika_reference',
    version='2019.11',
    author='Pavol Klacansky',
    author_email='klacansky@sci.utah.edu',
    description='Localized topological data analysis',
    url='https://github.com/klacansky/topologika',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    ext_modules=[module],
    setup_requires=['numpy>=1.13'],
    install_requires=['numpy>=1.13'],
)