from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import os
import configparser
import json
import redis
from src.train import MultiModel
from src.predict import Predictor
from fastapi.encoders import jsonable_encoder
from src.logger import Logger

app = FastAPI()

# Инициализация кастомного логгера
custom_logger_instance = Logger(show=True)  # show=True — вывод в консоль
logger = custom_logger_instance.get_logger("AppLogger")  # Получаем логгер

# Инициализация Redis
def create_redis_client():
    try:
        rc = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        # check connection/auth
        rc.ping()
        return rc
    except Exception as e:
        logger.warning(f"Redis not available at startup, continuing without cache: {e}")
        return None

redis_client = create_redis_client()

@app.post("/train/")
async def train_model(
    model_type: str = "d_tree",
    use_config: bool = True,
    save_model: bool = True,
    # Параметры для Logistic Regression
    solver: str = "lbfgs",
    max_iter: int = 100,
    # Параметры для Random Forest
    n_estimators: int = 100,
    criterion: str = "entropy",
    # Параметры для Decision Tree
    max_depth: int = 10,
    min_samples_split: int = 2,
    predict_flag: bool = False
):
    try:
        multi_model = MultiModel()
        
        if model_type == "log_reg":
            result = multi_model.log_reg(
                use_config=use_config, 
                solver=solver, 
                max_iter=max_iter, 
                predict=predict_flag,
                save=save_model
            )
        elif model_type == "rand_forest":
            result = multi_model.rand_forest(
                use_config=use_config, 
                n_estimators=n_estimators, 
                criterion=criterion, 
                predict=predict_flag,
                save=save_model
            )
        elif model_type == "d_tree":
            result = multi_model.d_tree(
                use_config=use_config, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                predict=predict_flag,
                save=save_model
            )
        elif model_type == "gnb":
            result = multi_model.gnb(predict=predict_flag, save=save_model)
        else:
            raise HTTPException(status_code=400, detail=f"Неизвестный тип модели: {model_type}")
        
        return {
            "model_trained": result, 
            "model_type": model_type,
            "model_saved": save_model
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict_model(mode: str = "smoke", file: UploadFile = None):
    cache_key = f"predict:{mode}"

    # try read from cache (keeps your existing robust catches)
    try:
        if redis_client is not None and redis_client.exists(cache_key):
            raw = redis_client.get(cache_key)
            if raw:
                try:
                    parsed = json.loads(raw)
                except Exception:
                    logger.warning("Не удалось распарсить данные из Redis, игнорируем кэш")
                    parsed = None
                if parsed is not None:
                    return {"from_cache": True, **parsed}
    except redis.exceptions.RedisError as re:
        logger.warning(f"Redis error while checking cache (treat as cache miss): {re}")
    except Exception as e:
        logger.warning(f"Unexpected error during Redis cache check: {e}")

    try:
        predictor = Predictor()

        if mode == "upload":
            if file is None:
                raise HTTPException(status_code=400, detail="Файл не предоставлен для режима 'upload'")
            file_contents = await file.read()
            result = predictor.predict_upload(file_contents)
        elif mode == "smoke":
            result = predictor.predict()
        else:
            raise HTTPException(status_code=400, detail="Неверный режим. Используйте 'smoke' или 'upload'")

        # make JSON-serializable
        safe_result = jsonable_encoder(result)

        # try save to redis (best-effort)
        try:
            if redis_client is not None:
                redis_client.set(cache_key, json.dumps(safe_result))
        except redis.exceptions.RedisError as re:
            logger.warning(f"Redis error while storing cache (ignored): {re}")
        except Exception as e:
            logger.warning(f"Unexpected error while writing to Redis cache: {e}")

        return safe_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    config = configparser.ConfigParser()
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', "config.ini")
    config.read(config_path, encoding="utf-8")
    try:
        host = config["FASTAPI"]["host"]
        port = config.getint("FASTAPI", "port")
    except KeyError:
        raise ValueError("В config.ini отсутствует секция [FASTAPI] или ключи host/port")
    uvicorn.run(app, host=host, port=port)