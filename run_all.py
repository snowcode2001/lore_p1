"""Process all conversations in l_conv.json

Usage:
    python run_all.py                  # single label (default)
    python run_all.py --multi-label    # multi label mode
"""

import sys
import json
import re
from app.providers.models import LocalModelProvider
from app.providers.storage import JSONFileStorage
from app.analyzer import BELIEF_CATEGORIES, BELIEF_PATTERN

multi_label = "--multi-label" in sys.argv
output_file = "data/history_multi.json" if multi_label else "data/history.json"

storage = JSONFileStorage(output_file)
models = LocalModelProvider()

with open("l_conv.json") as f:
    conversations = json.load(f)

all_beliefs = []
for i, conv in enumerate(conversations):
    messages = conv.get("messages_list", [])
    user_messages = [m for m in messages if m.get("ref_user_id") != 1]
    
    if not user_messages:
        print(f"[{i+1}/{len(conversations)}] 0 beliefs")
        continue
    
    user_id = user_messages[0].get("ref_user_id")
    beliefs = []
    
    for msg in user_messages:
        text = msg.get("message", "")
        sentences = re.split(r"[.!?]+", text)
        for s in sentences:
            s = s.strip()
            if s and BELIEF_PATTERN.search(s):
                result = models.classify_belief(s, BELIEF_CATEGORIES, multi_label=multi_label)
                embedding = models.get_embedding(s)
                beliefs.append({
                    "text": s,
                    "category": result["label"],
                    "category_confidence": result["score"],
                    "category_scores": result["all_scores"],
                    "embedding": embedding,
                })
                all_beliefs.append({"user_id": user_id, "category": result["label"]})
    
    storage.save(user_id, beliefs)
    print(f"[{i+1}/{len(conversations)}] {len(beliefs)} beliefs")

users = set(b["user_id"] for b in all_beliefs)
print(f"\n{len(conversations)} conversations, {len(users)} users, {len(all_beliefs)} beliefs")

categories = {}
for b in all_beliefs:
    categories[b["category"]] = categories.get(b["category"], 0) + 1
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print(f"\nSaved to {output_file}")
