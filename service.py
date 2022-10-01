from fastapi import FastAPI, Response
from pydantic import BaseModel
from src.prediction import predict

app = FastAPI()


class Info(BaseModel):
    image: str


@app.get("/status")
def index():
    return {"Service API OK"}


@app.post("/angle")
async def angle_corretion(data: Info, response: Response):
    angle = float(predict(data.image))
    response.status = 200
    print(angle)
    return {'angle': angle}
