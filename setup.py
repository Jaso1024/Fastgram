import sys
from setuptools import Extension, setup


class _PybindInclude:
    def __str__(self):
        import pybind11

        return pybind11.get_include()


def _extra_compile_args():
    if sys.platform == "win32":
        return ["/std:c++20", "/O2"]
    if sys.platform == "darwin":
        return ["-std=c++20", "-O3", "-DNDEBUG", "-mmacosx-version-min=10.15"]
    return ["-std=c++20", "-O3", "-DNDEBUG"]


def _extra_link_args():
    if sys.platform == "darwin":
        return ["-mmacosx-version-min=10.15"]
    return []


ext_modules = [
    Extension(
        "gram.cpp_engine",
        [
            "gram/cpp_engine.cpp",
            "src/engine.cc",
            "src/mmap_file.cc",
            "src/thread_pool.cc",
        ],
        include_dirs=[_PybindInclude(), "include"],
        language="c++",
        extra_compile_args=_extra_compile_args(),
        extra_link_args=_extra_link_args(),
    )
]


setup(
    name="gram",
    version="0.1.0",
    packages=["gram"],
    package_data={"gram": ["py.typed", "index_catalog.json"]},
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.11",
    install_requires=["transformers>=4.0"],
)
