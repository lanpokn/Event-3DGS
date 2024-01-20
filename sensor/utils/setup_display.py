from setuptools import setup, Extension
import numpy
import cv2

# These paths should be changed to your own paths
module = Extension("sgrb2lum",
                   sources=["D:/2023/computional imaging/my_pbrt/IEBCS-main/cpp/utils/display.cpp"],
                   include_dirs=[numpy.get_include(), "C:/Users//Desktop/code/opencv/build/include"],
                   library_dirs=["C:/Users//Desktop/code/opencv/build/x64/vc16/bin", "C:/Users//Desktop/code/opencv/build/x64/vc16/lib"],
                   libraries=["opencv_world480"],
                  #  extra_compile_args=["/std:c++11"],
                  #  extra_link_args=["/NODEFAULTLIB:MSVCRT", "/NODEFAULTLIB:LIBCMT"],
                   language="c++")

setup(
    name="sgrb2lum",
    version="1.0",
    description="Converts SRGB to luminance",
    ext_modules=[module]
)