from pydantic import BaseModel

class RiskRequest(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_count: float
    Value_sum: float
    Value_mean: float
    Value_std: float
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str

class RiskResponse(BaseModel):
    risk_probability: float