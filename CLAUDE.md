# Ajman Business Survival Score

Hackathon project for the Ajman Business Hackathon (Department of Economic Development, Ajman).
Challenge: *Developing Economic Forecasting Models and Innovative Statistical Equations — estimate future economic indicators and business activity trends.*

## Goal
Predict how likely a new or existing Ajman business is to survive 1, 3 and 5 years, and express it as an
explainable **Business Survival Score (0–100)**. Serve two users:
- **Entrepreneurs**: check an idea (sector, area, capital, size) before licensing.
- **DED analysts**: see which sectors/areas are oversaturated or high-risk, and forecast license cancellations.

## Core framing
This is a **survival analysis** problem, not plain classification.
- "Birth" = license issue date. "Death" (event) = license cancelled or not renewed within grace period.
- Businesses still active at the data cutoff are **right-censored** — never drop them or treat them as survivors-with-certainty.
- Duration = months from issue to event or to cutoff.

## The equation (our original contribution)

```
Score(t, x) = 100 × S0(t) ^ exp(β·x + γR·R + γD·D + γS·Saturation)
```

- `S0(t)`: baseline survival curve of the Cox model, horizon t (default 36 months).
- `β·x`: firm basics (sector, capital, employees, free zone, premises, legal form).
- `R`: **Reputation**, Bayesian-average Google rating (see Digital Presence module).
- `D`: **Digital Presence Index** 0–1 with learned weights.
- `Saturation`: demand-based market saturation 0–1 (see below).
- All γ are estimated jointly with β (hazard ratios + p-values). Score = 100 × P(still operating at t).
- History: v1 used a multiplicative penalty `× (1 − λ·Saturation)`; putting saturation inside the
  model scored better on validation/test, so the penalty is kept only as a reported comparison.

## Market saturation (src/features/saturation.py)
Saturation = competitors per customer, not share of businesses.
- Demand: resident population + visitors (`data/geo/ajman_areas.csv`, ESTIMATES: replace with
  Ajman Statistics Center data). Visitor weight v_s per sector (tourism 3.0, F&B 1.0, retail 0.4 …).
- Catchment: supply and demand smoothed across areas with K(d) = exp(−d / 2 km).
- `ratio = (supply/demand in catchment) / (Ajman average)`; `g(r) = clip((r − 0.75)/2.25, 0, 1)`.
- Local vs B2B: `Saturation = ℓ_s·g(ratio) + (1−ℓ_s)·0.2·g(LQ)`; ℓ_s = local share of customers
  (F&B 1.0 … manufacturing 0.1). Industrial clusters are agglomeration, not saturation.
- "Best areas" only suggests areas already holding ≥2% of the sector's licenses (zoning proxy).

## Data

### Preferred: real DED data (ask organizers)
Expected columns in `data/raw/`:
`license_id, issue_date, cancel_date (nullable), last_renewal_date, sector, activity_code, area,
license_type, legal_form, capital_aed, employees, free_zone (bool), premises_type`

### Fallback: synthetic data
`src/data/generate_synthetic.py` creates ~20,000 businesses, 2015–2025:
- Sectors: retail, F&B, construction, trading, professional services, manufacturing, tech, real estate, tourism.
- Ajman areas: Al Nuaimiya, Al Rashidiya, Al Jurf, Al Rawda, Al Mowaihat, Al Hamidiya, Ajman Industrial Area, Al Zorah, Masfout, Manama.
- Realistic hazards: retail and F&B fail faster; professional services and manufacturing slower; higher capital and
  more employees lower hazard; COVID-2020 shock raises hazard for F&B/tourism/retail; saturation raises hazard.
- Set a fixed random seed. Label clearly in UI and pitch as **synthetic** — never present it as real.

### Optional external signals
UAE open data (Bayanat), FCSC, UAE Central Bank, World Bank: GDP growth, inflation, oil price at founding year.

## Digital Presence module (social media & marketing activity)
Online activity is a strong success signal, so it becomes a covariate in the Cox model and a separate
**Digital Presence Index (D, 0–1)** shown in the UI.

### Signals
`has_website, google_rating, google_review_count, instagram_followers, posts_per_month,
engagement_rate, days_since_last_post, has_tiktok, has_whatsapp_business, responds_to_reviews`

### Reputation R (separate from marketing)
```
bayes = (n·rating + k·μ) / (n + k)    μ = Ajman average rating, k = 10
R     = clip(bayes − μ, training range)
```
Few reviews → rating barely counts; hundreds of reviews → rating counts fully.

### D formula
```
D = Σ w_i · norm(c_i)
c_posts     = log(1 + posts_per_month)
c_audience  = log(1 + followers × engagement_rate)     # engaged audience (bought followers don't help)
c_followers = log(1 + followers)                         # raw reach; learned weight ≈ 0
c_reviews   = log(1 + n_reviews × (rating − 1)/4)        # POSITIVE review volume
c_recency   = exp(−days_since_last_post / 30)
c_website   = has_website
```
Weights `w` are learned (Cox coefficients on the components, rescaled to sum to 1), not hand-picked.

### Leakage rule (important)
Failing businesses stop posting, so recent activity "predicts" failure trivially (reverse causality).
For training, use only signals measured in the **first 6 months after licensing**. Flag this in the Method tab.

### Data sources (legal and ToS-compliant only)
- **Synthetic** (default for hackathon): generator adds the signals above, correlated with survival.
- **Google Places API**: rating, review count, website, opening status — official API, needs a key.
- **Instagram Graph API – Business Discovery**: followers, media count, recent posts of public business accounts;
  needs our own Instagram Business account + Meta app. No scraping of Instagram/TikTok (violates ToS).
- **Deep research agent** (`src/research/digital_agent.py`): calls the Claude API with the web search tool to
  research one named business and return structured JSON (website found, platforms found, approx. followers,
  last activity date, review summary, marketing observations, source URLs). Used on demand in the dashboard,
  rate-limited, with results cached and sources shown. Never collect personal data about owners or staff.

### Dashboard additions
- Entrepreneur tab: inputs for planned marketing (posts/month, platforms) → "what-if" showing how score changes.
- "Research a business" box: enter name → agent report + D score + updated survival score.
- Map layer: average D per area.
- Entrepreneur tab: what-if charts for rating (5/50/300 reviews) and posting; engagement-rate input.

## Interactive map
- `src/viz/risk_map.py`: Folium via `streamlit-folium`, OpenStreetMap tiles (no API key).
- Map tab layers: average survival score, saturation of a chosen sector, overall saturation,
  expected closures (12 m), net new licenses (12 m), digital presence, market-gap labels, businesses (clustered).
- Area popup = full DED profile: residents, active businesses, saturation, top-3 crowded sectors,
  market gap, new/closed licenses, expected closures.
- DED tab: single-metric map with legend (saturation, competitors per 10k residents, closures,
  net entry, businesses per 10k, avg score) + heatmap + area profile table.
- Area centroids/population are approximate; verify before demo.

## Fairness & ethics
- Do NOT use owner nationality, gender, age or any personal attribute as features.
- Score is decision support, not a licensing gate. State this in the UI.

## Tech stack
- Python 3.11, pandas, numpy
- `lifelines` (Kaplan-Meier, CoxPH), `scikit-survival` (optional, for survival forests)
- `xgboost` (binary 3-year survival baseline for comparison)
- `shap` for explanations
- `streamlit` + `plotly` for the dashboard; `folium` or plotly map for area saturation
- `pytest` for tests

## Project structure
```
data/raw/            # input data (gitignored if real)
data/processed/
notebooks/           # EDA only, not production logic
src/data/            # generate_synthetic.py, load.py, clean.py
src/features/        # build_features.py, saturation.py
src/models/          # km.py, cox.py, xgb_baseline.py, score.py, evaluate.py
app/streamlit_app.py
tests/
models/              # saved fitted models (joblib)
README.md
```

## Modeling steps (do in order, one at a time)
1. Generate or load data → clean → compute `duration_months`, `event` columns.
2. EDA: Kaplan-Meier curves per sector and per area; log-rank tests.
3. Feature engineering: log(capital), employee bands, one-hot sector/area, founding-year macro vars, saturation.
4. Fit Cox PH model; check proportional hazards assumption (`check_assumptions`); fix violations by stratifying.
5. Fit XGBoost baseline for 3-year survival (exclude censored-before-36-months rows from that target only).
6. Fit λ for the saturation term on validation set.
7. Evaluate: concordance index (C-index), time-dependent Brier score, calibration plot at 12/36/60 months.
   Use a **time-based split** (train on earlier cohorts, test on later) — no random leakage.
8. Save models; write `score.py` exposing `predict_score(business_dict, horizon_months=36)`.

## Dashboard (Streamlit)
- **Entrepreneur tab**: inputs (sector, area, capital, employees, free zone, license type) → score gauge,
  survival curve with confidence band, top 3 risk/protective factors (SHAP), "similar businesses" survival rate.
- **DED tab**: heatmap of saturation by sector × area, map of Ajman areas colored by avg risk,
  forecast of expected cancellations next 12 months per sector (sum of predicted hazards over active licenses).
- **Method tab**: the equation rendered with LaTeX, plain-language explanation, model metrics, data disclaimer.

## Working rules for Claude Code
- Plan before coding each step; wait for approval on the plan.
- Keep functions small and typed; docstrings on every public function.
- Every model metric must be computed by code, never estimated.
- Add a pytest for the score function (bounds 0–100, monotonic: higher saturation → lower score).
- Keep README updated with how to run: `pip install -r requirements.txt`, `python -m src.data.generate_synthetic`,
  `python -m src.models.cox`, `streamlit run app/streamlit_app.py`.

## Pitch notes
- Problem: many new Ajman businesses close early; DED and founders lack forward-looking risk signals.
- Novelty: survival analysis + saturation-adjusted score, fully explainable.
- Impact: smarter licensing guidance, targeted SME support, early warning on oversaturated sectors.
- Next steps with real DED data: retrain, add rent and footfall data, integrate into the licensing portal.
