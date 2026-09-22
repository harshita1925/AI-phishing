"""
feature_schema.py

KNOWLEDGE BASE
==============
Declarative domain knowledge only -- no reasoning logic lives here.

This module stores the ten website features identified as the top
phishing indicators in the paper "AI-Based Phishing Detection Using
Machine Learning and Root Cause Analysis" (Figure 2 / Section V-B), plus
a human-readable "root cause" explanation for each one. The Root Cause
Analysis module (model_engine.py) looks up these explanations whenever
it needs to tell a user *why* a website was flagged -- it never invents
wording on the fly.

Each feature is encoded exactly as in the UCI Phishing Websites dataset:
    -1  ->  feature indicates the site is LEGITIMATE
     0  ->  feature is NEUTRAL / SUSPICIOUS (borderline)
     1  ->  feature indicates the site is PHISHING
"""

from typing import Dict, List

# Ordered by the importance ranking reported in the paper's Figure 2
FEATURES: List[str] = [
    "SSLfinal_State",
    "URL_of_Anchor",
    "web_traffic",
    "having_Sub_Domain",
    "Links_in_tags",
    "Prefix_Suffix",
    "SFH",
    "Request_URL",
    "Links_pointing_to_page",
    "Domain_registeration_length",
]

# Short label for the UI (feature name -> plain-English question)
FEATURE_QUESTIONS: Dict[str, str] = {
    "SSLfinal_State": "Does the site use a valid, trusted SSL/HTTPS certificate?",
    "URL_of_Anchor": "Do the links (anchors) on the page point to the site's own domain?",
    "web_traffic": "Does the site have an established web-traffic / popularity ranking?",
    "having_Sub_Domain": "Does the URL avoid excessive or unusual sub-domains?",
    "Links_in_tags": "Do most <meta>/<script>/<link> tags point to the site's own domain?",
    "Prefix_Suffix": "Is the domain free of a '-' (hyphen) prefix/suffix trick?",
    "SFH": "Does the login form submit (SFH) to the site's own domain?",
    "Request_URL": "Are images/media requested from the site's own domain?",
    "Links_pointing_to_page": "Do a healthy number of external sites link to this page?",
    "Domain_registeration_length": "Was the domain registered a long time ago (not brand new)?",
}

# The Root Cause Analysis "knowledge base": what it MEANS when a feature
# fires as a phishing signal (value == 1). This is the text the
# Explainable-AI layer surfaces to the user.
ROOT_CAUSE_EXPLANATIONS: Dict[str, str] = {
    "SSLfinal_State": "Invalid, missing, or self-signed SSL certificate -- no trusted HTTPS.",
    "URL_of_Anchor": "Anchor-tag mismatch: link text names one domain but points to another.",
    "web_traffic": "Very low or no web-traffic ranking -- the site is brand new / unknown.",
    "having_Sub_Domain": "Suspicious use of multiple sub-domains to imitate a trusted brand.",
    "Links_in_tags": "Large share of page tags load resources from an unrelated domain.",
    "Prefix_Suffix": "Domain uses a '-' prefix/suffix, a classic brand-impersonation trick.",
    "SFH": "Server Form Handler is blank or submits data to a different domain.",
    "Request_URL": "Images / objects on the page are fetched from an external domain.",
    "Links_pointing_to_page": "Unusually few external sites link to this page (low trust signal).",
    "Domain_registeration_length": "Domain was registered very recently or expires very soon.",
}


def get_all_features_summary() -> List[str]:
    """Return the full Knowledge Base as printable lines (KB reference view)."""
    return [
        f"**{f}** -- {FEATURE_QUESTIONS[f]}  \n  *If flagged (1):* {ROOT_CAUSE_EXPLANATIONS[f]}"
        for f in FEATURES
    ]
