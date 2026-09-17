# consciousLinux CUDA 启动环境修复

已确认编辑器直接执行 consciousLinux/bin/python -m ipykernel_launcher，绕过 kernelspec 的 argv/env，且继承 /etc/profile 设置的旧 CUDA 库目录。仅修改内核配置与 Conda 激活脚本没有覆盖这个入口。

新增环境内的 conscious_cuda_kernel_bootstrap.pth 和 _conscious_cuda_kernel_bootstrap.py，仅在使用 -m ipykernel_launcher 启动时生效。在加载 torch 之前纠正 LD_LIBRARY_PATH，并用相同参数、相同 PID 重新启动一次新生 Python 进程，让动态库加载器读取正确路径。已有正确路径时不会重新启动；普通 Python 命令不受影响。

安装位置：/home/p/anaconda3/envs/consciousLinux/lib/python3.11/site-packages/。
卸载：删除上述两个新增文件。先前的激活脚本、默认/专用内核修复独立保留。

验证使用一个全新临时 ipykernel，故意携带 /usr/local/cuda/lib64 旧路径，直接启动 Python，未使用修复后的 kernelspec。通过 Jupyter 消息执行 cuDNN 检查及 GPU 卷积，并从 /proc/self/maps 核对实际动态库来源。结果见 direct_kernel_verified.json。临时内核已关闭，用户原有内核未操作。

已经加载旧 cuDNN 的用户内核仍必须重启。无需再次安装 torch、修改系统 CUDA 或更改训练代码。
