#include "rasterize_points.h"
#include <torch/script.h>
#include <memory>
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>
#include <torch/torch.h>
#include <filesystem>

void printTensorInfo(const torch::Tensor &tensor, const bool print_data = false)
{
    std::cout << "Tensor size, type, device: " << tensor.sizes() << " " << tensor.dtype() << " " << tensor.device() << std::endl;
    if (print_data)
    {
        std::cout << "Tensor data: " << std::endl << tensor << std::endl;
    }
}

std::vector<char> get_the_bytes(std::string filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));
    input.close();
    return bytes;
}

torch::Tensor load_tensor(std::string filename)
{
    std::vector<char> f = get_the_bytes(filename);
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor my_tensor = x.toTensor();
    return my_tensor;
}

template <typename T>
T load_scalar(std::string filename)
{
    std::vector<char> f = get_the_bytes(filename);
    torch::IValue x = torch::pickle_load(f);
    T my_scalar = x.toScalar().to<T>();
    return my_scalar;
}

void runForward(const std::filesystem::path args_path)
{
    const torch::Tensor background = load_tensor(args_path / "bg.pt").to(torch::kCUDA);
    const torch::Tensor means3D = load_tensor(args_path / "means3D.pt").to(torch::kCUDA);
    const torch::Tensor colors = load_tensor(args_path / "colors_precomp.pt").to(torch::kCUDA);
    const torch::Tensor opacity = load_tensor(args_path / "opacities.pt").to(torch::kCUDA);
    const torch::Tensor scales = load_tensor(args_path / "scales.pt").to(torch::kCUDA);
    const torch::Tensor rotations = load_tensor(args_path / "rotations.pt").to(torch::kCUDA);
    const float scale_modifier = load_scalar<float>(args_path / "scale_modifier.pt");
    const torch::Tensor cov3D_precomp = load_tensor(args_path / "cov3Ds_precomp.pt").to(torch::kCUDA);
    const torch::Tensor viewmatrix = load_tensor(args_path / "viewmatrix.pt").to(torch::kCUDA);
    const torch::Tensor projmatrix = load_tensor(args_path / "projmatrix.pt").to(torch::kCUDA);
    // const float tanfovx = load_scalar<float>(args_path / "tanfovx.pt");
    // const float tanfovy = load_scalar<float>(args_path / "tanfovy.pt");
    const float tanfovx = 1.0;
    const float tanfovy = 0.5666666666666667;

    const int image_height = load_scalar<int>(args_path / "image_height.pt");
    const int image_width = load_scalar<int>(args_path / "image_width.pt");
    const torch::Tensor sh = load_tensor(args_path / "sh.pt").to(torch::kCUDA);
    const int sh_degree = load_scalar<int>(args_path / "sh_degree.pt");
    const torch::Tensor campos = load_tensor(args_path / "campos.pt").to(torch::kCUDA);
    const bool prefiltered = load_scalar<bool>(args_path / "prefiltered.pt");
    const bool debug = load_scalar<bool>(args_path / "debug.pt");          

    RasterizeGaussiansCUDA(
        background,
        means3D,
        colors,
        opacity,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        image_height,
        image_width,
        sh,
        sh_degree,
        campos,
        prefiltered,
        debug);
}


void runBackward(const std::filesystem::path args_path)
{
    const torch::Tensor background = load_tensor(args_path / "bg.pt").to(torch::kCUDA);
    const torch::Tensor means3D = load_tensor(args_path / "means3D.pt").to(torch::kCUDA);
    const torch::Tensor radii = load_tensor(args_path / "radii.pt").to(torch::kInt32).to(torch::kCUDA);
    const torch::Tensor colors = load_tensor(args_path / "colors_precomp.pt").to(torch::kCUDA);
    const torch::Tensor scales = load_tensor(args_path / "scales.pt").to(torch::kCUDA);
    const torch::Tensor rotations = load_tensor(args_path / "rotations.pt").to(torch::kCUDA);
    const float scale_modifier = load_scalar<float>(args_path / "scale_modifier.pt");

    const torch::Tensor cov3D_precomp = load_tensor(args_path / "cov3Ds_precomp.pt").to(torch::kCUDA);
    const torch::Tensor viewmatrix = load_tensor(args_path / "viewmatrix.pt").to(torch::kCUDA);
    const torch::Tensor projmatrix = load_tensor(args_path / "projmatrix.pt").to(torch::kCUDA);  

    const float tanfovx = 1.0;
    const float tanfovy = 0.5666666666666667;

    const torch::Tensor grad_color = load_tensor(args_path / "grad_out_color.pt").to(torch::kCUDA);
    const torch::Tensor grad_depth = load_tensor(args_path / "grad_depth.pt").to(torch::kCUDA);
    const torch::Tensor alpha = load_tensor(args_path / "alpha.pt").to(torch::kCUDA);
    const torch::Tensor grad_alpha = load_tensor(args_path / "grad_out_alpha.pt").to(torch::kCUDA);
    const torch::Tensor sh = load_tensor(args_path / "sh.pt").to(torch::kCUDA);
    const int sh_degree = load_scalar<int>(args_path / "sh_degree.pt");
    const torch::Tensor campos = load_tensor(args_path / "campos.pt").to(torch::kCUDA);
    const torch::Tensor geomBuffer = load_tensor(args_path / "geomBuffer.pt").to(torch::kCUDA);
    const int num_rendered = load_scalar<int>(args_path / "num_rendered.pt");
    const torch::Tensor binningBuffer = load_tensor(args_path / "binningBuffer.pt").to(torch::kCUDA);
    const torch::Tensor imgBuffer = load_tensor(args_path / "imgBuffer.pt").to(torch::kCUDA);
    const bool debug = load_scalar<bool>(args_path / "debug.pt");
    torch::Tensor dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations;

    printTensorInfo(viewmatrix, true);
    std::tie(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations) = RasterizeGaussiansBackwardCUDA(
        background,
        means3D,
        radii,
        colors,
        scales,
        rotations,
        scale_modifier,
        cov3D_precomp,
        viewmatrix,
        projmatrix,
        tanfovx,
        tanfovy,
        grad_color,
        grad_depth,
        grad_alpha,
        sh,
        sh_degree,
        campos,
        geomBuffer,
        num_rendered,
        binningBuffer,
        imgBuffer,
        alpha,
        debug);
}


int main()
{
    std::filesystem::path args_path = "/home/vy/projects/gaussian-rasterizer/test_data";
    runForward(args_path / "forward_tensors");
    runBackward(args_path / "backward_tensors");
    return 0;
}