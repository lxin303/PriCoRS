import time
import torch
from ptflops import get_model_complexity_info

def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def get_peak_mem_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)

@torch.no_grad()
def measure_inference_time(model, inputs, iters=50, warmup=10):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = model(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms_img = (t1 - t0) * 1000.0 / (iters * inputs.shape[0])
    return ms_img

def measure_flops(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        flops = FlopCountAnalysis(model, (inputs,)).total()
        return {"flops": float(flops), "backend": "fvcore"}
    except Exception as e:
        print(f"FLOPs failed: {repr(e)}")
        return None



    
