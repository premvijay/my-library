#include <iostream>

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include <Python.h>

namespace py = pybind11;

// float W_ngp(float diff){
// return 1;
// };

float W_cic(float diff){
    float d = std::abs(diff);
    if (d < 1){
        return 1.0 - d;
    } else {
        return 0;
    };
};

float W_tsc(float diff){
    float d = std::abs(diff);
    if (d < 0.5){
        return (0.75 - d*d);
    } else if (d < 1.5){
        return 0.5 * std::pow(1.5 - d, 2);
    } else {
        return 0;
    };
};

float W_sph_cubic_3d(float q){
    if (q < 1){
        return (1/M_PI) * ( 0.25* std::pow(2-q, 3) - std::pow(1-q, 3) );
    } else if (q < 2){
        return (1/M_PI) * 0.25* std::pow(2-q, 3);
    } else {
        return 0;
    };
};





size_t circ_index(float index, int N){
    if (index>=0 && index<N){ 
        return (size_t) index;
    } else if (index<0){
        return (size_t) (index + N);
    } else {
        return (size_t) (index - N);
    }
}


py::array_t<double> assign_ngp ( py::array_t<float> pos, int grid_size) {
    auto pos_unckd = pos.unchecked<2>();
    py::size_t num_prtcl = pos.shape(0);

    py::array_t<double> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = pos_unckd(i,0), yi = pos_unckd(i,1), zi = pos_unckd(i,2);
        float xif = std::floor(xi), yif = std::floor(yi), zif = std::floor(zi);

        size_t xc = circ_index(xif,grid_size), yc = circ_index(yif,grid_size), zc = circ_index(zif,grid_size);
        // float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

        grid_mut_unckd(xc,yc,zc) += 1;
    }
    return grid;
}


py::array_t<double> assign_cic ( py::array_t<float> pos, int grid_size) {
    auto pos_unckd = pos.unchecked<2>();
    py::size_t num_prtcl = pos.shape(0);

    py::array_t<double> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = pos_unckd(i,0), yi = pos_unckd(i,1), zi = pos_unckd(i,2);
        float xir = std::round(xi), yir = std::round(yi), zir = std::round(zi);
        float xid = xi-xir+0.5, yid = yi-yir+0.5, zid = zi-zir+0.5;  // distance to use in W() for u=v=w=0

        for (int u=-1; u < 1; u++){
            for (int v=-1; v < 1; v++){
                for (int w=-1; w < 1; w++){
                    size_t xc = circ_index(xir+u, grid_size), yc = circ_index(yir+v, grid_size),
                                         zc = circ_index(zir+w, grid_size);
                    // float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

                    grid_mut_unckd(xc,yc,zc) += W_cic(xid+u)* W_cic(yid+v)* W_cic(zid+w);
                }
            }
        }
    }
    return grid;
}


py::array_t<double> assign_tsc ( py::array_t<float> pos, int grid_size) {
    auto pos_unckd = pos.unchecked<2>();
    py::size_t num_prtcl = pos.shape(0);

    py::array_t<double> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = pos_unckd(i,0), yi = pos_unckd(i,1), zi = pos_unckd(i,2);
        float xif = std::floor(xi), yif = std::floor(yi), zif = std::floor(zi);
        float xid = xi-xif+0.5, yid = yi-yif+0.5, zid = zi-zif+0.5;  // distance to use in W() for u=v=w=0
        // py::print(xif,  yif, zif);

        for (int u=-1; u < 2; u++){
            for (int v=-1; v < 2; v++){
                for (int w=-1; w < 2; w++){
                    size_t xc = circ_index(xif+u, grid_size), yc = circ_index(yif+v, grid_size),
                                         zc = circ_index(zif+w, grid_size);
                    // float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

                    // py::print(xcc,  ycc, zcc);

                    grid_mut_unckd(xc,yc,zc) += W_tsc(xid+u)* W_tsc(yid+v)* W_tsc(zid+w);
                }
            }
        }
    }
    return grid;
}

py::array_t<double> assign_mass_cic ( py::array_t<float> pos, py::array_t<float> mass, int grid_size) {
    auto pos_unckd = pos.unchecked<2>();
    py::size_t num_prtcl = pos.shape(0);

    auto mass_unckd = mass.unchecked<1>();

    py::array_t<double> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = pos_unckd(i,0), yi = pos_unckd(i,1), zi = pos_unckd(i,2), mi=mass_unckd(i);
        float xir = std::round(xi), yir = std::round(yi), zir = std::round(zi);
        float xid = xi-xir+0.5, yid = yi-yir+0.5, zid = zi-zir+0.5;  // distance to use in W() for u=v=w=0

        for (int u=-1; u < 1; u++){
            for (int v=-1; v < 1; v++){
                for (int w=-1; w < 1; w++){
                    size_t xc = circ_index(xir+u, grid_size), yc = circ_index(yir+v, grid_size),
                                         zc = circ_index(zir+w, grid_size);
                    // float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

                    grid_mut_unckd(xc,yc,zc) += mi* W_cic(xid+u)* W_cic(yid+v)* W_cic(zid+w);
                }
            }
        }
    }
    return grid;
}


py::array_t<double> assign_mass_sph ( py::array_t<float> pos, py::array_t<float> mass, py::array_t<float> hsml, int grid_size) {
    auto pos_unckd = pos.unchecked<2>();
    py::size_t num_prtcl = pos.shape(0);

    auto mass_unckd = mass.unchecked<1>();
    auto hsml_unckd = hsml.unchecked<1>();

    py::array_t<double> grid({grid_size,grid_size,grid_size},0);
    auto grid_mut_unckd = grid.mutable_unchecked<3>();

    for (int u=0; u < grid_size; u++){
        for (int v=0; v < grid_size; v++){
            for (int w=0; w < grid_size; w++){
                grid_mut_unckd(u,v,w) = 0;
            }
        }
    }
    // grid = {0};

    // V_cell = 

    for (py::size_t i = 0; i < num_prtcl; i++) {
        float xi = pos_unckd(i,0), yi = pos_unckd(i,1), zi = pos_unckd(i,2), mi=mass_unckd(i), hi=hsml_unckd(i);
        float xif = std::floor(xi), yif = std::floor(yi), zif = std::floor(zi);
        float xid = xi-xif+0.5, yid = yi-yif+0.5, zid = zi-zif+0.5;  // distance to use in W() for u=v=w=0

        int hir = std::round(hi);

        for (int u=-2*hir; u < 2*hir; u++){
            for (int v=-2*hir; v < 2*hir; v++){
                for (int w=-2*hir; w < 2*hir; w++){
                    size_t xc = circ_index(xif+u, grid_size), yc = circ_index(yif+v, grid_size),
                                         zc = circ_index(zif+w, grid_size);
                    // float xcc = xc + 0.5, ycc = yc + 0.5, zcc = zc + 0.5;

                    float q = std::sqrt( (xid+u)*(xid+u) + (yid+v)*(yid+v) + (zid+w)*(zid+w) ) / hi;

                    grid_mut_unckd(xc,yc,zc) += mi* W_sph_cubic_3d(q)/ std::pow(hi,3);
                    // py::print(W_sph_cubic_3d(q));
                }
            }
        }
    }
    return grid;
}



PYBIND11_MODULE(particles_to_grid, m) {
    m.doc() = "pybind11 based binding to C++ code for density assignment from particle positions based on CIC scheme. "; // optional module docstring

    m.def("assign_ngp", &assign_ngp, "Assign particles to grid from particle positions by nearest grid point scheme.");

    m.def("assign_cic", &assign_cic, "Assign particles to grid from particle positions by Cloud in cell scheme.");
            // py::arg("i"), py::arg("j")=0);

    m.def("assign_tsc", &assign_tsc, "Assign particles to grid from particle positions by triangularly shaped cloud scheme.");

    m.def("assign_mass_cic", &assign_mass_cic, "Assign varying masses of particles to grid using particle positions by Cloud in cell scheme.");

    m.def("assign_mass_sph", &assign_mass_sph, "Assign mass to grid for SPH using cubic spline");

    // m.def("circ_index", &circ_index);

    // m.def("arfn",&arfn, "pass an array");
}