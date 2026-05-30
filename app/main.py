"""
OasisAC ML Backend — замена облачного сервера oasisac.biz
Принимает float[8][60] фичи от плагина, возвращает confidence (0.0–1.0)
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import time
import logging
import os
import numpy as np

from .model import predict, load_model

logger = logging.getLogger("oasis-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="OasisAC ML Backend", version="1.0.0")

# Загружаем модель при старте
@app.on_event("startup")
async def startup():
    import threading
    threading.Thread(target=load_model, daemon=True).start()
    logger.info("OasisAC backend started, model loading in background...")


# ── Схема запроса (точно как плагин отправляет) ────────────────────────────

class Features(BaseModel):
    delta_yaw:       list[float]   # 60 значений
    delta_pitch:     list[float]
    accel_yaw:       list[float]
    accel_pitch:     list[float]
    jerk_yaw:        list[float]
    jerk_pitch:      list[float]
    gcd_error_yaw:   list[float]
    gcd_error_pitch: list[float]

class AnalyzeRequest(BaseModel):
    player_uuid:  str
    player_name:  str
    timestamp:    int
    window_size:  int
    server_port:  Optional[int] = None
    features:     Features


# ── Эндпоинт /analyze ──────────────────────────────────────────────────────

API_KEY = os.getenv("OASIS_API_KEY", "your_secret_key_here")

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, x_api_key: Optional[str] = Header(None)):
    # Проверка ключа
    if x_api_key != API_KEY:
        logger.warning(f"Invalid API key from {req.player_name}: '{x_api_key}'")
        raise HTTPException(status_code=401, detail="Invalid API key")

    t0 = time.monotonic()

    # Собираем матрицу [8 x 60]
    f = req.features
    matrix = np.array([
        f.delta_yaw,
        f.delta_pitch,
        f.accel_yaw,
        f.accel_pitch,
        f.jerk_yaw,
        f.jerk_pitch,
        f.gcd_error_yaw,
        f.gcd_error_pitch,
    ], dtype=np.float32)

    confidence = predict(matrix)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        f"[{req.player_name}] confidence={confidence:.3f} "
        f"({confidence*100:.1f}%) time={elapsed_ms}ms"
    )

    return {"confidence": float(confidence), "time_ms": elapsed_ms}


@app.get("/health")
async def health():
    return {"status": "ok"}
