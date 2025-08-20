from distutils.core import setup  # 用于构建和安装Python模块
from distutils.extension import Extension  # 用于定义扩展模块
from Cython.Build import cythonize  # 用于将Cython代码编译为C扩展
import numpy  # 导入numpy以获取其包含目录

# 定义扩展模块列表
extensions = [
    Extension(
        "im2col_cython",  # 扩展模块的名称（导入时使用）
        ["im2col_cython.pyx"],  # 包含Cython源代码的文件
        include_dirs=[numpy.get_include()]  # 包含numpy的头文件目录（用于处理numpy数组）
    ),
]

# 编译扩展模块
# cythonize会将Cython代码转换为C代码，然后由distutils编译为二进制扩展
setup(ext_modules=cythonize(extensions),)
