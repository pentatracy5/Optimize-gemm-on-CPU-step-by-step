# Optimize-gemm-on-CPU-step-by-step

本项目是一系列在 CPU 上逐步优化通用矩阵乘法（GEMM）的参考实现，并附带与 Intel oneAPI Math Kernel Library (MKL) 参考实现的性能对比。

## 前置依赖

在编译之前，请确保你的系统已经安装以下工具/库：

- **CMake >= 3.18**
- 支持 **C++17** 的编译器（GCC、Clang 或 MSVC）
- **OpenMP** 运行库（通常随现代编译器一起提供）
- **Intel oneAPI MKL**（需要手动安装）

### 1. 安装 Intel oneAPI MKL

本项目使用 MKL 的 `cblas_sgemm` 作为参考实现，因此**必须先自行安装 Intel oneAPI MKL**。安装方式任选其一：

- 通过 [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) 完整安装
- 仅安装 [Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
- 使用包管理器（如 apt/yum/conda）安装 `intel-oneapi-mkl`

### 2. 设置 oneAPI 环境变量（setvars）

CMake 需要通过 `MKLConfig.cmake` 查找 MKL。安装 oneAPI 后，**每次打开新的终端都需要先执行 `setvars` 脚本**，让环境变量生效：

- **Linux / macOS：**
  ```bash
  source /path/to/oneapi/setvars.sh
  ```

- **Windows（cmd）：**
  ```cmd
  "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
  ```

- **Windows（PowerShell）：**
  ```powershell
  & "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
  ```

> 默认安装路径通常是 `/opt/intel/oneapi/setvars.sh`（Linux）或 `C:\Program Files (x86)\Intel\oneapi\setvars.bat`（Windows）。如果你的安装路径不同，请相应替换。

如果你不想每次执行 `setvars`，也可以在调用 CMake 时显式指定 MKL 的 CMake 配置目录：

```bash
cmake -B build -D MKL_DIR=/path/to/oneapi/mkl/latest/lib/cmake/mkl ..
```

## 编译

推荐在 Release 模式下编译以获得最佳性能：

```bash
# 1. 进入项目根目录并确保已执行 setvars
source /path/to/oneapi/setvars.sh  # Linux/macOS

# 2. 创建并进入构建目录
mkdir -p build && cd build

# 3. 生成构建系统（Release 模式）
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. 编译
cmake --build . -j$(nproc)
```

Windows 上建议使用 Visual Studio 生成器：

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

编译成功后会生成可执行文件：

- Linux/macOS：`build/src/matmul`
- Windows：`build\src\Release\matmul.exe`

## 运行

### 3. 将 MKL 动态库路径加入 PATH

编译只是链接了 MKL 的导入库。运行时系统还需要找到 MKL 的**动态库（DLL / .so / .dylib）**，否则程序会报错找不到 `mkl_*`、`libiomp5` 等库。因此，**在运行可执行文件之前，请把 MKL 动态库目录加入环境变量**：

- **Linux：**
  ```bash
  export LD_LIBRARY_PATH=/path/to/oneapi/mkl/latest/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/path/to/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
  ./src/matmul
  ```

- **macOS：**
  ```bash
  export DYLD_LIBRARY_PATH=/path/to/oneapi/mkl/latest/lib:$DYLD_LIBRARY_PATH
  export DYLD_LIBRARY_PATH=/path/to/oneapi/compiler/latest/lib:$DYLD_LIBRARY_PATH
  ./src/matmul
  ```

- **Windows（cmd）：**
  ```cmd
  set PATH=C:\path\to\oneapi\mkl\latest\bin;%PATH%
  set PATH=C:\path\to\oneapi\compiler\latest\bin;%PATH%
  src\Release\matmul.exe
  ```

- **Windows（PowerShell）：**
  ```powershell
  $env:PATH = "C:\path\to\oneapi\mkl\latest\bin;$env:PATH"
  $env:PATH = "C:\path\to\oneapi\compiler\latest\bin;$env:PATH"
  src\Release\matmul.exe
  ```

> 如果你是通过 `setvars.bat`/`setvars.sh` 设置的 oneAPI 环境，该脚本通常已经把 MKL 和编译器运行库路径加入环境变量；保险起见，仍建议检查上述路径是否已包含。

## 其他注意事项

1. **务必使用 Release 模式**：
   - Debug 模式不会开启 `-O3 / -O2` 和向量化优化，性能可能相差数倍甚至数十倍。
   - CMake 默认生成的是 Debug 配置（Windows 多配置生成器除外），请显式指定 `-DCMAKE_BUILD_TYPE=Release` 或在构建时使用 `--config Release`。

2. **CPU 特性**：
   - 部分实现使用 AVX2/FMA 指令集。默认 Release 编译会开启 `/arch:AVX2`（MSVC）或 `-march=native`（GCC/Clang）。
   - 如果你的 CPU 不支持 AVX2，可能需要在 `src/CMakeLists.txt` 中把 `/arch:AVX2` 或 `-march=native` 改成适合你 CPU 的选项。

3. **OpenMP 线程数**：
   - `include/config.h` 中定义了 `OMP_THREADS = 16`。你可以根据实际 CPU 核心数修改该值，或在运行前通过环境变量覆盖：
     ```bash
     export OMP_NUM_THREADS=8
     ./src/matmul
     ```

4. **调优参数**：
   - 分块大小（`GEMM_MC`、`GEMM_NC`、`GEMM_KC`、`GEMM_MR`、`GEMM_NR`）对性能影响很大。
   - 不同实现（`MatMul01` ~ `MatMul10`）可能需要不同的分块参数。运行前请确认 `include/config.h` 中启用的参数与你要测试的实现匹配。

5. **精度与基准**：
   - 程序会将自定义实现与 MKL 的 `cblas_sgemm` 结果对比，容忍误差为 `TOLERANCE`（默认约 `2e-5 * NREPEATS`）。
   - 如果改动过大导致误差超出容限，程序会报错。

6. **Windows 特别提示**：
   - 使用 MSVC 时建议从 **「x64 Native Tools Command Prompt」** 启动终端，这样 CMake 能正确找到 64 位编译器。
   - 如果在 PowerShell 中 `setvars.bat` 没有生效，请使用 `cmd.exe /k` 方式或参考 Intel 官方文档使用 PowerShell 模块。
