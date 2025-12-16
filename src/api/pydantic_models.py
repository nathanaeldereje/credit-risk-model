from pydantic import BaseModel, Field

class CreditRiskRequest(BaseModel):
    # Numeric Features (Aggregated)
    total_transactions: int = Field(..., description="Total number of transactions")
    total_amount: float = Field(..., description="Sum of absolute transaction amounts")
    avg_amount: float = Field(..., description="Average transaction amount")
    std_amount: float = Field(..., description="Standard deviation of transaction amounts")
    avg_fee_paid: float = Field(..., description="Average fee paid (Value - Amount)")
    total_refunds_count: int = Field(..., description="Count of negative transactions")
    tx_hour_mean: float = Field(..., description="Average hour of transaction (0-23)")
    tx_day_mean: float = Field(..., description="Average day of month (1-31)")
    
    # Categorical Features (Modes)
    ProviderId: str = Field(..., description="Most frequent Provider ID")
    ProductCategory: str = Field(..., description="Most frequent Product Category")
    ChannelId: str = Field(..., description="Most frequent Channel ID")
    PricingStrategy: str = Field(..., description="Most frequent Pricing Strategy")

    class Config:
        json_schema_extra = {
            "example": {
                "total_transactions": 25,
                "total_amount": 500000.0,
                "avg_amount": 20000.0,
                "std_amount": 5000.0,
                "avg_fee_paid": 500.0,
                "total_refunds_count": 0,
                "tx_hour_mean": 14.5,
                "tx_day_mean": 15.0,
                "ProviderId": "ProviderId_6",
                "ProductCategory": "financial_services",
                "ChannelId": "ChannelId_3",
                "PricingStrategy": "PricingStrategy_2"
            }
        }

class CreditRiskResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
    credit_score: int