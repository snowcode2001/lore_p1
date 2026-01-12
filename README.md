# Belief Evaluation API

Extracts and analyzes user beliefs from conversations using NLP models.

## Setup

```bash
# Create and activate virtual environment

# if you have pyenv
pyenv virtualenv 3.11 belief-eval-api-venv
pyenv activate belief-eval-api-venv

# if you dont, assuming you have python set up under the python keyword
python -m venv belief-eval-api-venv
"belief-eval-api-venv/Scripts/activate"

# Install dependencies
# this can take a few minutes, especially because of torch, so get a coffee or snack
pip install -r requirements.txt
```

## Download Models

```bash
python download_models.py
```

This downloads models (~1.5GB) to `~/.cache/huggingface/`.

## Start Server

```bash
uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`.

## Process All Conversations (pre-populate history storage)

```bash
python run_all.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/evaluate-beliefs` | Analyze conversation |
| GET | `/api/v1/history/{user_id}` | Get user's belief history |
| GET | `/api/v1/history/{user_id}/?store=sentiment` | Get user's sentiment history |
| GET | `/api/v1/history/{user_id}/?store=risk` | Get user's risk history |

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/evaluate-beliefs \
  -H "Content-Type: application/json" \
  -d "$(cat l_conv.json | jq '.[0]')"
```

Input format: see `l_conv.json`

## Example Response

```json
{
  "conversation_id": 98696,
  "user_id": 782,
  "belief_count": 5,
  "beliefs": [
    {
      "text": "I believe honesty is the foundation of everything",
      "category": "core_values",
      "category_confidence": 0.64,
      "embedding": [0.01, -0.02, ...]
    },
    {
      "text": "I believe neighbors should look out for each other",
      "category": "social_beliefs",
      "category_confidence": 0.59,
      "embedding": [0.03, -0.01, ...]
    }
  ],
  "downstream_outputs": {
    "value_attribution": {
      "self_beliefs": [
        {"text": "I feel like I'm too old to learn new tricks"}
      ],
      "self_belief_count": 2
    },
    "storybot": {
      "dominant_theme": "social_beliefs",
      "themes": ["self_efficacy", "core_values", "social_beliefs"]
    },
    "content_recommendation": {
      "topic_affinities": ["self_efficacy", "core_values", "social_beliefs"]
    }
  }
}
```

## Tests

```bash
pytest tests/ -v
```

## Design Choices

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT: l_conv.json                                                          │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ messages_list: [{ref_user_id, message, timestamp}, ...]                 │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PROCESSING                                                                  │
│  1. Filter user messages (exclude bot)                                      │
│  2. Extract belief sentences (regex patterns)                               │
│  3. Classify each belief (zero-shot → category)                             │
│  4. Generate embedding (MiniLM → vector)                                    │
|  5. Score overall sentiment                                                 |
|  6. Score risk topics                                                       |
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐───────────────────────┐
          ▼                       ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ VALUE ATTRIBUTION │   │ STORYBOT          │   │ CONTENT RECO      │   │ Score Storage     │
├───────────────────┤   ├───────────────────┤   ├───────────────────┤   ├───────────────────┤
│ self_beliefs:     │   │ dominant_theme    │   │ topic_affinities  │   | sentiment_score   |
│   [{text}]        │   │ themes: [...]     │   │                   │   | risk_scores       |
│ self_belief_count │   │                   │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘   └───────────────────┘
```

### Models

| Component | Model | Why |
|-----------|-------|-----|
| Belief classification | `facebook/bart-large-mnli` (zero-shot) | No training needed; define categories at runtime |
| Embeddings | `all-MiniLM-L6-v2` | Fast, small; enables similarity search on beliefs |
| Sentiment | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | Fast, somewhat small; allows gradual positive minus negative |
| Risk Scoring | `facebook/bart-large-mnli` (zero-shot) (multi_label=True) | No training needed; define categories at runtime |

### Storage

| Environment | Implementation | Why |
|-------------|----------------|-----|
| Tests | `MockStorage` | Fast, no cleanup needed |
| Tests | `MockGenericStorage` | Fast, no cleanup needed |
| Prototype | `JSONFileStorage` | Human-readable, easy to share |

### Belief Detection

Beliefs are identified by regex patterns matching phrases like:
- "I believe", "I feel", "I think", "I value"
- "I firmly believe", "I've come to believe"

### Belief Categories (defined in `analyzer.py`):
- `self_efficacy`: beliefs about one's own capabilities
- `core_values`: fundamental values/priorities
- `social_beliefs`: beliefs about human connection/society
- `institutional_trust`: trust/distrust of organizations
- `technology_stance`: views on technology's role

### Risk Categories (defined in `analyzer.py`):
- `self_harm`
- `violence`
- `depression`

### Production Path

1. **Data**: Prototype uses file input; production would consume JSON message streams
2. **Models**: Replace `LocalModelProvider` with hosted inference endpoints
3. **Belief detection**: Current regex catches explicit beliefs only; consider classifying all sentences
4. **Categories**: Replace ad-hoc categories with a validated taxonomy (e.g., Schwartz Values, Moral Foundations) or custom definition
5. **Storage**: Consider more edge cases like inserting duplicate message logs
6. **Messages**: Label indivudal messaegs with an ID instead of just timestamp
7. **Data Visualization**: Set up plot of sentiment and risk factors over time by user
8. **Potential Topic Extraction**: Extract potential topics from conversation text itself