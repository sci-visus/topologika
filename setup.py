import platform
import setuptools

# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class GetNumpyInclude(object):
    def __str__(self):
        import numpy
        return numpy.get_include()


if platform.system() == 'Windows':
    compile_args = []
    link_args = []
elif platform.system() == 'Linux':
    compile_args = ['-std=c99']
    link_args = []
elif platform.system() == 'Darwin':
    compile_args = ['-std=c99']
    link_args = []
else:
    raise Exception('Only Windows, Linux, and Mac is supported')

module = setuptools.Extension(
    'topologika',
    sources=['topologikamodule.c', 'binding.cpp'],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    include_dirs=[GetNumpyInclude()]
)

setuptools.setup(
    name='topologika',
    version='2019.11',
    author='Pavol Klacansky',
    author_email='klacansky@sci.utah.edu',
    description='Localized topological data analysis',
    url='https://github.com/pavolklacansky/topologika',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    ext_modules=[module],
    setup_requires=['numpy>=1.13'],
    install_requires=['numpy>=1.13'],
)
