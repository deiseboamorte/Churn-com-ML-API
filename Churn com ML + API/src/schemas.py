from pydantic import BaseModel

class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str