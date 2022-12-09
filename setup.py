import platform
import setuptools

# https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy
class GetNumpyInclude(object):
    def __str__(self):
        import numpy
        return numpy.get_include()


if platform.system() == 'Windows':
    compile_args = ['/openmp']
    link_args = []
elif platform.system() == 'Linux':
    compile_args = ['-std=c99', '-fopenmp']
    link_args = ['-fopenmp']
else:
    raise Exception('Only Windows and Linux is supported')

module = setuptools.Extension(
    'topologika',
    sources=['topologikamodule.c'],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    include_dirs=[GetNumpyInclude()]
)

setuptools.setup(
    name='topologika',
    version='2022.12',
    author='Pavol Klacansky',
    author_email='pavol@klacansky.com',
    description='Localized topological data analysis',
    url='https://github.com/sci-visus/topologika',
    packages=setuptools.find_packages(),
    python_requires='>=3.11',
    ext_modules=[module],
    setup_requires=['numpy>=1.13'],
    install_requires=['numpy>=1.13'],
)
