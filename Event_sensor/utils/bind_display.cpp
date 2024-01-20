#include "Camera.h"
namespace py = pybind11;
extern  std::vector<float> emptyWave;
// bind camera api
void bind_display_api(py::module_ & e2e)
{
    // bind display base
    py::class_<Display> disp(e2e, "Display");
    // bind display srgb
    py::class_<DisplaySRGB> dispSRGB(e2e, "DisplaySRGB");
    dispSRGB.def("__repr__", []( DisplaySRGB &a){return "<Display to rendering srgb image to multispectral photons>";})
            .def(py::init< std::vector<float>&,  std::vector<cv::Point3f>&,  std::vector<cv::Point3f>&>())
            .def("rendering", &DisplaySRGB::rendering, "Rendering srgb image to SceneSpectral", py::arg("rgb"), py::arg("waveDst")=emptyWave)
            .def_property_readonly("chroma", &DisplaySRGB::getChroma, "Get the color space of the display")
            .def_property_readonly("maxLuminance", &DisplaySRGB::getMaxLuminance, "The maximum luminance of the display")
            ;
    // bind natureSRGB
    extern  DisplaySRGB natureSRGB;
    e2e.attr("natureSRGB") = natureSRGB;
}