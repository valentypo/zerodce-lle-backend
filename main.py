"""
Zero-DCE Real-Time Enhancement Server
FastAPI WebSocket server for streaming low-light image enhancement
"""

import os
import sys
import torch
import cv2
import base64
import numpy as np
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from model import DCENet, load_model


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
PROCESS_SIZE = 512  # Must be multiple of 32 for Zero-DCE architecture

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL MODEL (loaded once at startup)
# ============================================================================

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager: load model on startup, cleanup on shutdown.
    """
    global model
    
    # STARTUP
    logger.info(f"Starting Zero-DCE Server | Device: {DEVICE}")
    logger.info(f"Model path: {MODEL_PATH}")
    
    # Verify model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    
    # Load model
    try:
        model = load_model(MODEL_PATH, device=DEVICE)
        logger.info(f"✓ Model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
    
    yield
    
    # SHUTDOWN
    logger.info("Shutting down Zero-DCE Server")
    if model is not None:
        del model
        torch.cuda.empty_cache()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Zero-DCE Real-Time Enhancement API",
    description="WebSocket-based low-light image enhancement server",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration - Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def base64_to_image(base64_str: str) -> np.ndarray:
    """
    Convert base64 encoded string to OpenCV image array.
    
    Args:
        base64_str: Base64 encoded image string (JPEG, PNG, etc)
    
    Returns:
        Image as numpy array (BGR format)
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_str)
        
        # Decode JPEG/PNG bytes to image
        image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image_array is None:
            raise ValueError("Failed to decode image")
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise


def image_to_base64(image_array: np.ndarray, format: str = 'JPEG') -> str:
    """
    Convert OpenCV image array to base64 encoded string.
    
    Args:
        image_array: Image as numpy array (BGR format)
        format: Image format - 'JPEG' or 'PNG'
    
    Returns:
        Base64 encoded image string with data URI prefix
    """
    try:
        # Encode image to bytes
        if format.upper() == 'JPEG':
            success, encoded = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            success, encoded = cv2.imencode('.png', image_array)
        
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        image_bytes = encoded.tobytes()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        
        # Add data URI prefix
        data_uri = f"data:image/jpeg;base64,{base64_str}"
        
        return data_uri
    
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise


def enhance_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    Apply Zero-DCE enhancement to image.
    
    Args:
        image_bgr: Input image in BGR format (OpenCV)
    
    Returns:
        Enhanced image in BGR format
    """
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Store original size
        original_h, original_w = image_rgb.shape[:2]
        
        # Resize to processing size (multiple of 32)
        image_resized = cv2.resize(image_rgb, (PROCESS_SIZE, PROCESS_SIZE))
        
        # Normalize to [0, 1]
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        
        # Add channel and batch dimensions: (H, W, 3) -> (1, 3, H, W)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(DEVICE)
        
        # Inference (no gradients needed)
        with torch.no_grad():
            enhanced_tensor = model(image_tensor)
        
        # Post-process: convert back to numpy
        enhanced_np = enhanced_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        
        # Denormalize: [0, 1] -> [0, 255]
        enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
        
        # Resize back to original dimensions
        enhanced_resized = cv2.resize(enhanced_np, (original_w, original_h))
        
        # Convert RGB back to BGR for OpenCV compatibility
        enhanced_bgr = cv2.cvtColor(enhanced_resized, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr
    
    except Exception as e:
        logger.error(f"Error during enhancement: {str(e)}")
        raise


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint. Returns server status and device info.
    """
    return JSONResponse({
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": model is not None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    })


@app.get("/info")
async def model_info():
    """
    Get model information.
    """
    return JSONResponse({
        "model_name": "Zero-DCE",
        "description": "Zero-Reference Deep Curve Estimation for real-time low-light image enhancement",
        "input_size": (PROCESS_SIZE, PROCESS_SIZE),
        "input_channels": 3,
        "output_channels": 3,
        "device": DEVICE,
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0
    })


# ============================================================================
# WEBSOCKET ENDPOINT - REAL-TIME STREAMING
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected | Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected | Active connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {str(e)}")


manager = ConnectionManager()


@app.websocket("/ws/enhance")
async def websocket_enhance(websocket: WebSocket):
    """
    WebSocket endpoint for real-time image enhancement streaming.
    
    Expected message format from client:
    {
        "type": "frame",
        "image": "<base64-encoded-image>"
    }
    
    Response format:
    {
        "type": "enhanced",
        "image": "<base64-encoded-enhanced-image>",
        "processing_time_ms": <float>
    }
    """
    await manager.connect(websocket)
    
    try:
        frame_count = 0
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") != "frame":
                continue
            
            frame_count += 1
            
            try:
                # Decode base64 image
                image_bgr = base64_to_image(data.get("image", ""))
                
                # Enhanced image
                import time
                start_time = time.time()
                enhanced_bgr = enhance_image(image_bgr)
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Encode enhanced image
                enhanced_base64 = image_to_base64(enhanced_bgr)
                
                # Send response
                response = {
                    "type": "enhanced",
                    "image": enhanced_base64,
                    "processing_time_ms": processing_time,
                    "frame_count": frame_count
                }
                
                await websocket.send_json(response)
                
                # Log stats every 30 frames
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames | Last frame: {processing_time:.2f}ms")
            
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                # Send error response
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected after {frame_count} frames")
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
