from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='trees_layer',
      ext_modules=[cpp_extension.CppExtension('trees_layer', ['trees_layer.cpp'])],
      include_dirs=cpp_extension.include_paths(),
      cmdclass={'build_ext': cpp_extension.BuildExtension})