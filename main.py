"""
Zero-DCE Real-Time Enhancement Server - GPU OPTIMIZED
FastAPI WebSocket server with maximum GPU utilization
"""

import os
import torch
import cv2
import base64
import numpy as np
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from model import DCENet, load_model

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
PROCESS_SIZE = 512

# GPU Optimization settings
BATCH_SIZE = 4  # Process multiple frames at once
USE_TORCH_COMPILE = hasattr(torch, 'compile')  # PyTorch 2.0+ optimization
USE_CUDNN_BENCHMARK = True
PREFETCH_FRAMES = 2  # Number of frames to prefetch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL RESOURCES
# ============================================================================

model = None
thread_pool = ThreadPoolExecutor(max_workers=4)  # For async CPU operations

# Preallocate GPU memory for faster transfers
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    # STARTUP
    logger.info(f"🚀 Starting GPU-Optimized Zero-DCE Server | Device: {DEVICE}")
    
    if DEVICE == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    try:
        model = load_model(MODEL_PATH, device=DEVICE)
        
        # Convert to half precision for faster inference
        if DEVICE == 'cuda':
            model = model.half()
        
        model.eval()  # Set to evaluation mode
        
        # Compile model for faster inference (PyTorch 2.0+)
        if USE_TORCH_COMPILE and DEVICE == 'cuda':
            logger.info("🔥 Compiling model with torch.compile...")
            model = torch.compile(model, mode='max-autotune')
        
        # Warmup: Run a few dummy inferences to initialize CUDA kernels
        logger.info("Warming up GPU...")
        dummy_input = torch.randn(1, 3, PROCESS_SIZE, PROCESS_SIZE, 
                                   device=DEVICE, dtype=torch.half if DEVICE == 'cuda' else torch.float)
        with torch.inference_mode():
            for _ in range(5):
                _ = model(dummy_input)
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        
        logger.info(f"✅ Model ready | Warmup complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
    
    yield
    
    # SHUTDOWN
    logger.info("Shutting down...")
    thread_pool.shutdown(wait=True)
    if model is not None:
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Zero-DCE GPU-Optimized API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GPU-OPTIMIZED IMAGE PROCESSING
# ============================================================================

def base64_to_numpy(base64_str: str) -> np.ndarray:
    """Fast base64 decode to numpy array."""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    image_bytes = base64.b64decode(base64_str)
    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if image_array is None:
        raise ValueError("Failed to decode image")
    
    return image_array


def numpy_to_base64(image_array: np.ndarray) -> str:
    """Fast numpy to base64 encode."""
    success, encoded = cv2.imencode('.jpg', image_array, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])  # Slightly lower quality for speed
    
    if not success:
        raise ValueError("Failed to encode image")
    
    image_bytes = encoded.tobytes()
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_str}"


# Reusable GPU tensors (avoid allocations)
class TensorCache:
    def __init__(self):
        self.cache = {}
    
    def get_tensor(self, shape, dtype, device):
        key = (shape, dtype, device)
        if key not in self.cache:
            self.cache[key] = torch.empty(shape, dtype=dtype, device=device)
        return self.cache[key]

tensor_cache = TensorCache()


def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """Optimized preprocessing with minimal CPU overhead."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_h, original_w = image_rgb.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image_rgb, (PROCESS_SIZE, PROCESS_SIZE), 
                               interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor efficiently
    image_tensor = torch.from_numpy(image_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Transfer to GPU with pinned memory for faster transfer
    dtype = torch.half if DEVICE == 'cuda' else torch.float
    image_tensor = image_tensor.to(DEVICE, dtype=dtype, non_blocking=True)
    
    return image_tensor, (original_h, original_w)


def postprocess_image(enhanced_tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
    """Optimized postprocessing."""
    original_h, original_w = original_size
    
    # Convert back to numpy (keep on GPU as long as possible)
    enhanced_np = enhanced_tensor.squeeze(0).cpu().float().permute(1, 2, 0).numpy()
    enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
    
    # Resize back to original
    if (enhanced_np.shape[0] != original_h) or (enhanced_np.shape[1] != original_w):
        enhanced_np = cv2.resize(enhanced_np, (original_w, original_h), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # Convert RGB to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
    
    return enhanced_bgr


async def enhance_image_async(image_bgr: np.ndarray) -> np.ndarray:
    """Async wrapper for GPU enhancement."""
    loop = asyncio.get_event_loop()
    
    # Run preprocessing in thread pool
    image_tensor, original_size = await loop.run_in_executor(
        thread_pool, preprocess_image, image_bgr
    )
    
    # GPU inference (blocking but fast)
    with torch.inference_mode():
        enhanced_tensor = model(image_tensor)
    
    # Run postprocessing in thread pool
    enhanced_bgr = await loop.run_in_executor(
        thread_pool, postprocess_image, enhanced_tensor, original_size
    )
    
    return enhanced_bgr


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0-gpu-optimized"}


@app.get("/health")
async def health_check():
    health_data = {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if DEVICE == 'cuda':
        health_data.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "gpu_memory_cached_gb": torch.cuda.memory_reserved() / 1e9,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        })
    
    return JSONResponse(health_data)


@app.get("/info")
async def model_info():
    return JSONResponse({
        "model_name": "Zero-DCE (GPU Optimized)",
        "input_size": (PROCESS_SIZE, PROCESS_SIZE),
        "device": DEVICE,
        "optimizations": {
            "fp16": DEVICE == 'cuda',
            "torch_compile": USE_TORCH_COMPILE and DEVICE == 'cuda',
            "cudnn_benchmark": USE_CUDNN_BENCHMARK,
            "async_processing": True,
            "thread_pool_workers": thread_pool._max_workers
        },
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0
    })


# ============================================================================
# WEBSOCKET - GPU STREAMING
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ Client connected | Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"❌ Client disconnected | Total: {len(self.active_connections)}")


manager = ConnectionManager()


@app.websocket("/ws/enhance")
async def websocket_enhance(websocket: WebSocket):
    await manager.connect(websocket)
    
    frame_count = 0
    total_time = 0
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") != "frame":
                continue
            
            frame_count += 1
            
            try:
                import time
                start_time = time.perf_counter()
                
                # Decode in thread pool
                loop = asyncio.get_event_loop()
                image_bgr = await loop.run_in_executor(
                    thread_pool, base64_to_numpy, data.get("image", "")
                )
                
                # Enhance (GPU accelerated)
                enhanced_bgr = await enhance_image_async(image_bgr)
                
                # Encode in thread pool
                enhanced_base64 = await loop.run_in_executor(
                    thread_pool, numpy_to_base64, enhanced_bgr
                )
                
                processing_time = (time.perf_counter() - start_time) * 1000
                total_time += processing_time
                
                response = {
                    "type": "enhanced",
                    "image": enhanced_base64,
                    "processing_time_ms": round(processing_time, 2),
                    "frame_count": frame_count
                }
                
                await websocket.send_json(response)
                
                # Log stats
                if frame_count % 30 == 0:
                    avg_time = total_time / frame_count
                    fps = 1000 / avg_time if avg_time > 0 else 0
                    logger.info(f"📊 Frames: {frame_count} | Avg: {avg_time:.2f}ms | FPS: {fps:.1f}")
                    
                    if DEVICE == 'cuda' and frame_count % 60 == 0:
                        logger.info(f"🎮 GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            
            except Exception as e:
                logger.error(f"❌ Frame {frame_count} error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        avg_fps = frame_count / (total_time / 1000) if total_time > 0 else 0
        logger.info(f"📊 Session ended | {frame_count} frames | Avg FPS: {avg_fps:.1f}")
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
        manager.disconnect(websocket)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1  # Single worker for GPU efficiency
    )