from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.providers.storage import JSONFileStorage
from app.providers.models import LocalModelProvider
from app.analyzer import BeliefAnalyzer

app = FastAPI(
    title="Belief Evaluation API",
    description="Extracts and analyzes user beliefs from conversations",
)

storage = JSONFileStorage("data/belief-history-testing.json")
risk_storage = JSONFileStorage("data/risk-history-testing.json")
sentiment_storage = JSONFileStorage("data/sentiment-history-testing.json")
models = LocalModelProvider()
analyzer = BeliefAnalyzer(models, storage, risk_storage, sentiment_storage)


class Message(BaseModel):
    ref_conversation_id: int
    ref_user_id: int
    transaction_datetime_utc: str
    screen_name: str
    message: str


class Conversation(BaseModel):
    messages_list: list[Message]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/evaluate-beliefs")
def evaluate_beliefs(conversation: Conversation):
    result = analyzer.analyze_conversation(conversation.model_dump())
    return result


@app.get("/api/v1/history/{user_id}/") # use alternative store with /?store=risk at the end
def get_user_history(user_id: int, store: str = "beliefs"):
    if store == "beliefs":
        history = storage.get_history(user_id)
    elif store == "risk":
        history = risk_storage.get_history(user_id)
    elif store == "sentiment":
        history = sentiment_storage.get_history(user_id)
    else:
        return {"user_id": user_id, "error": f"No storage for {store}, perhaps there's a typo."}
        
    return {"user_id": user_id, "history": history, "entry_count": len(history)}
