from fastapi import UploadFile
from pydantic import BaseModel


class TrainRequest(BaseModel):
    name: str
    image: UploadFile
