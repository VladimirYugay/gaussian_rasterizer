# General Info
This repo contains modification of the original differential Gaussian rasterizer. The main difference are the depth backpropagation, and alpha rendering. Also, we provide some test data and cmake file which allows debugging the CUDA code itself.

# Install
This command will install the repo to a python project:
```
pip install .
```

# Build and debug
If you want to debug the cuda code with the debugger, we provide some test data for it.

### Download test data
```
cd test_data
git clone https://huggingface.co/datasets/voviktyl/gaussian-rasterizer-test-data
```
All the test data was obtained from Replica dataset. Adjust the input paths and process the downloaded dump files with the command:
```
python convert_dump.py
```

This would unpack the dump files to <i>.pt</i> tensors. 


### Build the project
First, adjust the paths cmake file. After that run:
```
mkdir build
cd build
cmake ..
make
```
This would create `rasterizer` executable which can be examined with the VSCode debugger.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>