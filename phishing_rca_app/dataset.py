"""
dataset.py

PROBLEM FORMULATION / DATA LAYER
=================================
Builds the training data and defines the preset demo cases used by the
app. This mirrors the paper's Section III-A/B: 30-feature UCI Phishing
Websites Dataset, -1/1 labelling, 80/20 train-test split (the split
itself happens in model_engine.py).

IMPORTANT -- data source note:
This environment has no internet access to the original UCI repository
(https://archive.ics.uci.edu/ml/datasets/Phishing+Websites), so
load_dataset() below GENERATES a synthetic dataset that:
  - uses the exact same 10 top features and -1/0/1 encoding as the
    real UCI dataset,
  - weights SSLfinal_State and URL_of_Anchor most heavily when
    generating labels, matching the paper's actual Figure 2 findings,
  - adds random noise so the classification task is non-trivial.

To use the real dataset for your submission, download the UCI CSV and
replace load_dataset() with:
    return pd.read_csv("phishing.csv")
Everything downstream (train/test split, Random Forest, feature
importance / RCA) works unchanged either way, because it only depends
on the column names in feature_schema.FEATURES + a "Result" column.
"""

import numpy as np
import pandas as pd

from feature_schema import FEATURES

RANDOM_STATE = 42

# Importance weights used only to GENERATE realistic synthetic labels.
# These mirror the ranking from the paper's Root Cause Analysis (Fig. 2):
# SSL and anchor-tag mismatch dominate, the rest matter less.
_LABEL_WEIGHTS = {
    "SSLfinal_State": 3.0,
    "URL_of_Anchor": 2.4,
    "web_traffic": 1.3,
    "having_Sub_Domain": 1.1,
    "Links_in_tags": 0.9,
    "Prefix_Suffix": 0.8,
    "SFH": 0.7,
    "Request_URL": 0.6,
    "Links_pointing_to_page": 0.5,
    "Domain_registeration_length": 0.4,
}


def load_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """Return a DataFrame with columns FEATURES + 'Result' (1=phishing, -1=legit)."""
    rng = np.random.default_rng(RANDOM_STATE)
    data = {f: rng.choice([-1, 0, 1], size=n_samples, p=[0.45, 0.15, 0.40]) for f in FEATURES}
    df = pd.DataFrame(data)

    score = sum(df[f] * w for f, w in _LABEL_WEIGHTS.items())
    score = score + rng.normal(0, 1.5, size=n_samples)
    df["Result"] = np.where(score > np.median(score), 1, -1)
    return df


# ---------------------------------------------------------------------
# Preset demo cases shown in the sidebar ("Try a preset example")
# ---------------------------------------------------------------------
EXAMPLE_CASES = {
    "phishing_example": {
        "label": "Preset: Typical phishing site (bad SSL + anchor mismatch)",
        "features": {
            "SSLfinal_State": 1, "URL_of_Anchor": 1, "web_traffic": 1,
            "having_Sub_Domain": 1, "Links_in_tags": 0, "Prefix_Suffix": 1,
            "SFH": 0, "Request_URL": 0, "Links_pointing_to_page": 0,
            "Domain_registeration_length": 1,
        },
    },
    "legitimate_example": {
        "label": "Preset: Typical legitimate site (valid SSL, matching links)",
        "features": {
            "SSLfinal_State": -1, "URL_of_Anchor": -1, "web_traffic": -1,
            "having_Sub_Domain": -1, "Links_in_tags": -1, "Prefix_Suffix": -1,
            "SFH": -1, "Request_URL": -1, "Links_pointing_to_page": -1,
            "Domain_registeration_length": -1,
        },
    },
    "borderline_example": {
        "label": "Preset: Borderline site (valid SSL, but anchor mismatch)",
        "features": {
            "SSLfinal_State": -1, "URL_of_Anchor": 1, "web_traffic": 0,
            "having_Sub_Domain": 1, "Links_in_tags": 0, "Prefix_Suffix": -1,
            "SFH": 0, "Request_URL": 0, "Links_pointing_to_page": -1,
            "Domain_registeration_length": 0,
        },
    },
}
