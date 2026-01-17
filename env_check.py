import sys
import platform
import torch

def check_environment():
    print("=" * 60)
    print(" Training Environment & Device Information ")
    print("=" * 60)

    # ---------- Python 环境 ----------
    print("\n[Python]")
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"Python Executable   : {sys.executable}")
    print(f"Operating System    : {platform.system()} {platform.release()}")
    print(f"Platform            : {platform.platform()}")

    # ---------- PyTorch ----------
    print("\n[PyTorch]")
    print(f"PyTorch Version     : {torch.__version__}")
    print(f"PyTorch Built w/ CUDA: {torch.version.cuda}")
    print(f"PyTorch Built w/ cuDNN: {torch.backends.cudnn.version()}")
    print(f"cuDNN Enabled       : {torch.backends.cudnn.enabled}")

    # ---------- CUDA Runtime ----------
    print("\n[CUDA Runtime]")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Runtime Version: {torch.version.cuda}")
        print(f"GPU Count           : {torch.cuda.device_count()}")
        print(f"Current Device ID   : {torch.cuda.current_device()}")
        print(f"Current Device Name : {torch.cuda.get_device_name(torch.cuda.current_device())}")

        # ---------- GPU Details ----------
        print("\n[GPU Details]")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Compute Capability : {props.major}.{props.minor}")
            print(f"  Total Memory       : {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors    : {props.multi_processor_count}")
    else:
        print("CUDA not available. Using CPU.")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_environment()
