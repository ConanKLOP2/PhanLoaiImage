from __future__ import annotations

import inspect


def main() -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime chua duoc cai.")
        return 1

    print("ONNX Runtime providers:")
    for provider in ort.get_available_providers():
        print(f"- {provider}")

    try:
        from nudenet import NudeDetector
    except ImportError:
        print("nudenet chua duoc cai.")
        return 1

    print(f"NudeDetector signature: {inspect.signature(NudeDetector)}")
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("GPU CUDA kha dung cho ONNX Runtime.")
    else:
        print("Chua thay CUDAExecutionProvider. Dang chay CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
