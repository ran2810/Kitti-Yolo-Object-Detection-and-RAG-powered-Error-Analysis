import torch
import subprocess
import time
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil
import matplotlib.pyplot as plt
import warnings

# Filter by the specific string fragment to avoid silencing other important warnings
warnings.filterwarnings("ignore", message=".*torch.Tensor inputs should be normalized.*")

# ------------------------------------------------------------
# FP32 Evaluation (CPU)
# ------------------------------------------------------------
def evaluate_fp32_cpu(model_path, data_yaml):
    model = YOLO(model_path)

    print("\n=== FP32 CPU Evaluation ===")
    metrics = model.val(data=data_yaml, device="cpu")
    map50 = metrics.box.map50
    map95 = metrics.box.map

    # Warm-up 
    for _ in range(5): 
        dummy = torch.randn(1, 3, 640, 640)
        model.predict(dummy, device="cpu")
    
    def bench_fp32():
        start = time.time()
        for _ in range(20):
            dummy = torch.randn(1, 3, 640, 640)
            _ = model.predict(dummy, device="cpu")
        return (time.time() - start) / 20

    latency = bench_fp32()

    return {
        "mAP50": map50, "mAP95": map95, "latency_ms": latency * 1000
    }

# ------------------------------------------------------------
# FP32 Evaluation (GPU)
# ------------------------------------------------------------
def evaluate_fp32_gpu(model_path, data_yaml):
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    model = YOLO(model_path)

    print("\n=== FP32 Evaluation (GPU) ===")
    metrics = model.val(data=data_yaml, device=0)
    map50 = metrics.box.map50
    map95 = metrics.box.map

    dummy = torch.randn(1, 3, 640, 640).cuda()

    # Warm-up
    for _ in range(5):
        model.predict(dummy, device=0)

    def bench():
        start = time.time()
        for _ in range(20):
            model.predict(dummy, device=0)
        return (time.time() - start) / 20

    latency = bench()

    return {"mAP50": map50, "mAP95": map95, "latency_ms": latency * 1000}

# ------------------------------------------------------------
# FP32 Evaluation (GPU - TensorRT)
# ------------------------------------------------------------
def evaluate_fp32_tensorrt(model_path, data_yaml):
    
    model = YOLO(model_path)

    print("\n=== FP32 TensorRT Evaluation ===")

    # Step 1: Export ONNX
    onnx_path = model.export(format="onnx", opset=12)
    engine_path = "model_fp32.trt"

    # Step 2: Build TensorRT engine
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=1024"
    ])

    # Step 3: Benchmark latency
    start = time.time()

    subprocess.run([
        "trtexec",
        f"--loadEngine={engine_path}",
        "--shapes=input:1x3x640x640",
        "--iterations=50",
        "--warmUp=10"
    ], stdout=subprocess.DEVNULL)

    end = time.time()
    latency = (end - start) / 50

    # Step 4: Accuracy (ONNX proxy)
    print("\n=== Accuracy Estimation (ONNX Proxy) ===")
    onnx_model = YOLO(onnx_path)

    metrics = onnx_model.val(
        data=data_yaml,
        imgsz=640,
        batch=1,
        device="cpu"
    )

    return {
        "mAP50": metrics.box.map50,
        "mAP95": metrics.box.map,
        "latency_ms": latency * 1000
    }

# FP16 CPU is not supported 

# ------------------------------------------------------------
# FP16 Evaluation (GPU)
# ------------------------------------------------------------
def evaluate_fp16_gpu(model_path, data_yaml):
    model = YOLO(model_path)

    print("\n=== FP16 Evaluation (GPU) ===")
    metrics = model.val(data=data_yaml, device=0, half=True)
    map50 = metrics.box.map50
    map95 = metrics.box.map

    dummy = torch.randn(1, 3, 640, 640).cuda().half()

    # Warm-up
    for _ in range(5):
        model.predict(dummy, device=0, half=True)

    def bench_fp16():
        start = time.time()
        for _ in range(20):
            model.predict(dummy, device=0, half=True)
        return (time.time() - start) / 20

    latency = bench_fp16()

    return {
        "mAP50": map50,
        "mAP95": map95,
        "latency_ms": latency * 1000
    }


# ------------------------------------------------------------
# FP16 Evaluation (GPU - TensorRT)
# ------------------------------------------------------------
def evaluate_fp16_tensorrt(model_path, data_yaml):
    model = YOLO(model_path)

    print("\n=== FP16 TensorRT Evaluation ===")

    # Step 1: Export ONNX
    onnx_path = model.export(format="onnx", opset=12)
    engine_path = "model_fp16.trt"

    # Step 2: Build TensorRT engine
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_path}",
        "--fp16",
        f"--saveEngine={engine_path}",
        "--workspace=1024"
    ])

    # Step 3: Benchmark latency
    start = time.time()

    subprocess.run([
        "trtexec",
        f"--loadEngine={engine_path}",
        "--shapes=input:1x3x640x640",
        "--iterations=50",
        "--warmUp=10"
    ], stdout=subprocess.DEVNULL)

    end = time.time()
    latency = (end - start) / 50

    # Step 4: Accuracy (ONNX proxy)
    print("\n=== Accuracy Estimation (ONNX Proxy) ===")
    onnx_model = YOLO(onnx_path)

    metrics = onnx_model.val(
        data=data_yaml,
        imgsz=640,
        batch=1,
        device="cpu"
    )

    return {
        "mAP50": metrics.box.map50,
        "mAP95": metrics.box.map,
        "latency_ms": latency * 1000
    }


# ------------------------------------------------------------
# INT8 PTQ Evaluation (CPU - ONNX)
# ------------------------------------------------------------
def evaluate_int8_cpu(model_path, data_yaml):
    model = YOLO(model_path)

    # print("\n=== INT8 CPU Quantization (PTQ) ===")
    onnx_path = model.export(format="onnx", opset=12)
    int8_path = "best_int8.onnx"

    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)

    session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    dummy = torch.randn(1, 3, 640, 640)
    dummy_np = dummy.numpy().astype(np.float32)

    def bench_int8():
        start = time.time()
        for _ in range(20):
            session.run(None, {input_name: dummy_np})
        return (time.time() - start) / 20

    latency = bench_int8()

    print("\n=== INT8 Evaluation (ONNX Runtime) ===")

    int8_model = YOLO("runs/detect/train/weights/best.onnx")

    int8_metrics = int8_model.val(
        data="kitti.yaml",
        imgsz=640,
        batch=1,
        device="cpu"
    )

    return {
        "mAP50": int8_metrics.box.map50,
        "mAP95": int8_metrics.box.map,
        "latency_ms": latency * 1000
    }

# INT8 GPU: CUDA kernels for INT8 inference are not implemented in PyTorch

# ------------------------------------------------------------
# INT8 PTQ Evaluation (GPU - TensorRT)
# ------------------------------------------------------------
def evaluate_int8_tensorrt(model_path, data_yaml):
    model = YOLO(model_path)

    print("\n=== INT8 TensorRT Evaluation ===")

    # Step 1: Export ONNX
    onnx_path = model.export(format="onnx", opset=12)
    engine_path = "model_int8.trt"

    # Step 2: Build TensorRT engine
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_path}",
        "--int8",
        "--fp16",
        f"--saveEngine={engine_path}",
        "--workspace=1024"
    ])

    # Step 3: Benchmark latency
    start = time.time()

    subprocess.run([
        "trtexec",
        f"--loadEngine={engine_path}",
        "--shapes=input:1x3x640x640",
        "--iterations=50",
        "--warmUp=10"
    ], stdout=subprocess.DEVNULL)

    end = time.time()
    latency = (end - start) / 50

    # Step 4: Accuracy (reuse ONNX as proxy)
    print("\n=== Accuracy Estimation (ONNX Proxy) ===")
    onnx_model = YOLO(onnx_path)

    metrics = onnx_model.val(
        data=data_yaml,
        imgsz=640,
        batch=1,
        device="cpu"
    )

    return {
        "mAP50": metrics.box.map50,
        "mAP95": metrics.box.map,
        "latency_ms": latency * 1000
    }

# ------------------------------------------------------------
# PLOTTING FUNCTION
# ------------------------------------------------------------
def plot_results(results):
    # Extract latency values
    labels = []
    latencies = []
    map50_vals = []
    map95_vals = []

    for key, val in results.items():
        if "latency_ms" in val:
            labels.append(key)
            latencies.append(val["latency_ms"])
            map50_vals.append(val.get("mAP50", None))
            map95_vals.append(val.get("mAP95", None))

    # -----------------------------
    # LATENCY PLOT
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(labels, latencies, color="skyblue")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Comparison Across Precisions & Backends")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # ACCURACY PLOT (mAP50)
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(labels, map50_vals, color="lightgreen")
    plt.ylabel("mAP50")
    plt.title("Accuracy Comparison (mAP50)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # ACCURACY PLOT (mAP95)
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(labels, map95_vals, color="salmon")
    plt.ylabel("mAP95")
    plt.title("Accuracy Comparison (mAP95)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# MASTER FUNCTION
# ------------------------------------------------------------
def trigger_all_benchmarks(model_path="best.pt", data_yaml="kitti.yaml"):

    results = {}

    # CPU
    results["FP32_CPU"] = evaluate_fp32_cpu(model_path, data_yaml)
    results["INT8_CPU"] = evaluate_int8_cpu(model_path, data_yaml)

    # GPU
    if torch.cuda.is_available():
        results["FP32_GPU"] = evaluate_fp32_gpu(model_path, data_yaml)
        results["FP16_GPU"] = evaluate_fp16_gpu(model_path, data_yaml)
    else:
        results["FP16_GPU"] = {"error": "CUDA not available"}

    # TensorRT
    trtexec_path = shutil.which("trtexec")
    if trtexec_path:
        results["FP32_TRT"] = evaluate_fp32_tensorrt(model_path, data_yaml)
        results["FP16_TRT"] = evaluate_fp16_tensorrt(model_path, data_yaml)
        results["INT8_TRT"] = evaluate_int8_tensorrt(model_path, data_yaml)

    print("\n==============================")
    print("     FINAL COMPARISON")
    print("==============================")
    for k, v in results.items():
        print(f"\n{k}: {v}")

    # plot results
    plot_results(results)

    return results


# ------------------------------------------------------------
# Run directly from CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    trigger_all_benchmarks("best.pt", "kitti.yaml")