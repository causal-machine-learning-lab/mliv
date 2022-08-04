import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="mliv",       # 模块名称
    version="0.0.2",                  # 当前版本
    author="anpeng wu",               # 作者
    author_email="anpwu2019@gmail.com",        # 作者邮箱
    description="machine learning for instrumental variable (IV) regression",    # 模块简介
    long_description=long_description,          # 模块详细介绍
    long_description_content_type="text/markdown",  # 模块详细介绍格式
    url="https://github.com/anpwu/mliv.git",               # 模块github地址
    packages=setuptools.find_packages(),        # 自动找到项目中导入的模块
    # 模块相关的元数据（更多的描述）
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # 依赖模块
    install_requires=[
        'argparse',
        'pillow',
        'numba',
        'cvxopt',
    ],
  # python版本
    python_requires=">=3.7",
)  