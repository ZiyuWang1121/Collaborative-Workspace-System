from __future__ import annotations

import itertools
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CANONICAL_SKILLS = [
    "python",
    "sql",
    "excel",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "ml",
    "machine learning",
    "nlp",
    "deep learning",
    "tensorflow",
    "pytorch",
    "data visualization",
    "plotly",
    "altair",
    "matplotlib",
    "seaborn",
    "project management",
    "agile",
    "scrum",
    "communication",
    "planning",
    "statistics",
    "regression",
    "classification",
    "time series",
    "cloud",
    "aws",
    "gcp",
    "azure",
]


def normalize_tokens(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    tokens = [t.strip().lower() for t in text.replace("/", " ").replace(",", " ").split()]
    tokens = [t for t in tokens if t]
    return tokens


def extract_skills_from_text(text: str) -> List[str]:
    tokens = normalize_tokens(text)
    text_join = " ".join(tokens)
    found = []
    for skill in CANONICAL_SKILLS:
        s_norm = skill.lower()
        if s_norm in text_join:
            found.append(s_norm)
    # if no skills detected, back off to frequent tech keywords from vocabulary
    if not found:
        vocab_hits = [t for t in tokens if t in set(CANONICAL_SKILLS)]
        found = list(dict.fromkeys(vocab_hits))  # unique preserve order
    # default minimum set for demo
    if not found:
        found = ["python", "sql"]
    return list(dict.fromkeys(found))


def build_team_index(team_df: pd.DataFrame) -> Dict[str, Any]:
    # Expect columns: name, role, skills
    skill_texts = team_df["skills"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", token_pattern=r"[A-Za-z\+\-#\.]+")
    matrix = vectorizer.fit_transform(skill_texts)
    return {
        "df": team_df.reset_index(drop=True),
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def _vectorize_skills(skills: List[str], vectorizer: TfidfVectorizer) -> np.ndarray:
    query = " ".join(skills)
    return vectorizer.transform([query])


def match_team_to_skills(team_index: Dict[str, Any], required_skills: List[str], k: int = 4) -> List[Dict[str, Any]]:
    df: pd.DataFrame = team_index["df"]
    vectorizer: TfidfVectorizer = team_index["vectorizer"]
    matrix = team_index["matrix"]
    q_vec = _vectorize_skills(required_skills, vectorizer)
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = np.argsort(-sims)[:k]
    members = []
    for i in top_idx:
        row = df.iloc[int(i)]
        members.append(
            {
                "name": row.get("name", f"Member {int(i)+1}"),
                "role": row.get("role", "Contributor"),
                "skills": row.get("skills", ""),
                "similarity": float(sims[int(i)]),
            }
        )
    return members


