import fastapi
from pydantic import BaseModel
from typing import List
import logging
import pandas as pd

from .model import DelayModel

logging.basicConfig(level=logging.DEBUG)


app = fastapi.FastAPI(debug=True)


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightList(BaseModel):
    flights: List[Flight]


delay_model = DelayModel()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(flight_list: FlightList) -> dict:
     try:
        data = pd.DataFrame([flight.dict() for flight in flight_list.flights])

        logging.debug(f"Incoming prediction request with data: {data}")

        features = delay_model.preprocess(data)
        predictions = delay_model.predict(features)

        logging.info(f"Predictions: {predictions}")

        return {"predict": predictions}
    except fastapi.HTTPException as e:
        logging.error(f"HTTP Exception: {e.detail}")
        raise
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise fastapi.HTTPException(
            status_code=500, detail="An error occurred during prediction."
        )
