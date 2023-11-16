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


# Adding a comment to check ci.workflow
@app.post("/predict", status_code=200)
async def post_predict(flight_list: FlightList) -> dict:
    try:
        data = pd.DataFrame([flight.dict() for flight in flight_list.flights])

        logging.debug(f"Incoming prediction request with data: {data}")

        known_categories = delay_model.category_mapping
        for column in ["OPERA", "TIPOVUELO", "MES"]:
            if any(
                item not in known_categories[column] for item in data[column].unique()
            ):
                unknown_categories = set(data[column].unique()) - set(
                    known_categories[column]
                )
                error_message = (
                    f"Unknown categories in column {column}: {unknown_categories}"
                )
                logging.error(error_message)
                raise fastapi.HTTPException(status_code=400, detail=error_message)

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
