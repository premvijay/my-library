#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <Python.h>

// #include <vector>

namespace py = pybind11;


py::array_t<bool> within_sphere ( py::array_t<float> posd, float cenx, float ceny, float cenz, float rad, float box_size) {
    auto posd_unckd = posd.unchecked<2>();
    py::size_t num_prtcl = posd.shape(0);

    py::array_t<bool> selection({num_prtcl},0);
    auto selection_mut_unckd = selection.mutable_unchecked<1>();

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float diffx = std::fabs(posd_unckd(i,0)-cenx), diffy = std::fabs(posd_unckd(i,1)-ceny), diffz = std::fabs(posd_unckd(i,2)-cenz);

        // diff = std::min(std::fabs(xi-cenx), box_size - std::fabs(xi-cenx))
        if (std::pow( std::min(diffx, box_size-diffx), 2) + std::pow( std::min(diffy, box_size-diffy), 2) + std::pow( std::min(diffz, box_size-diffz), 2) < rad*rad ) {
            selection_mut_unckd(i) = true;
        }  else {selection_mut_unckd(i) = false;}

    }

    return selection;

}


py::array_t<bool> within_cube ( py::array_t<float> posd, float cenx, float ceny, float cenz, float side, float box_size) {
    auto posd_unckd = posd.unchecked<2>();
    py::size_t num_prtcl = posd.shape(0);

    py::array_t<bool> selection({num_prtcl},0);
    auto selection_mut_unckd = selection.mutable_unchecked<1>();

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float diffx = std::fabs(posd_unckd(i,0)-cenx), diffy = std::fabs(posd_unckd(i,1)-ceny), diffz = std::fabs(posd_unckd(i,2)-cenz);

        if (std::min(diffx, box_size-diffx) < side/2 && std::min(diffy, box_size-diffy) < side/2 && std::min(diffz, box_size-diffz) < side/2 ) {
            selection_mut_unckd(i) = true;
        }  else {selection_mut_unckd(i) = false;}

    }

    return selection;

}



PYBIND11_MODULE(select_particles, m) {
    m.doc() = "pybind11 based binding to C++ code for density assignment from particle positions based on CIC scheme. "; // optional module docstring

    m.def("within_sphere", &within_sphere, "return indices of the particles that lie within a given sphere.");

    m.def("within_cube", &within_cube, "return indices of the particles that lie within a given cube.");
            // py::arg("i"), py::arg("j")=0);

    // m.def("arfn",&arfn, "pass an array");
}