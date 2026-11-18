# Ajman Digital Growth Platform

Built in about 2 hours at the one-day **Ajman Business Hackathon**, organised by the Ajman Department of Economic Development (DED) with Ajman Free Zone (FZA), and pitched in 10 minutes.

**What it does**
1. **Forecast:** projects Ajman trade licenses (new, exits, active) to 2028, by sector and by district, with a survival model behind the exits.
2. **Guide:** a Registration Advisor that, at licensing, suggests districts with less competition and higher forecast growth.
3. **Connect:** a Creator Hub and Growth Matchmaker that pair struggling businesses with Ajman-licensed social media creators.

Pitch deck: open [`docs/pitch_deck.html`](docs/pitch_deck.html) in a browser (arrow keys to navigate, F for fullscreen).

> **Data note:** aggregates come from Ajman open data (data.ajman.ae) and figures shared at the hackathon. Firm-level records are **simulated** and calibrated to those totals, and the marketing effect size is an **assumption**. The raw spreadsheets supplied by the organizers are not included in this repository.

---

Predicts how likely an Ajman business is to survive 1, 3 and 5 years, as an explainable 0–100 score:

```
Score(t, x) = 100 × S0(t)^exp(β·x + γR·Reputation + γD·DigitalPresence + γS·Saturation)
```

Cox survival model with a Bayesian-average Google reputation, a learned Digital Presence Index
(engaged audience, posting, positive reviews) and demand-based market saturation (competitors per
customer, with a 2 km catchment and local-vs-B2B sectors).
See `CLAUDE.md` for the full design.

> Real data: 79 district polygons, DED license counts per district x activity, yearly new/renewed/cancelled/active licenses (2016–2025), monthly free zone registrations, Ajman Chamber cancellations (see `src/data/real.py`). Firm-level records are simulated and calibrated to these figures.

## Run
The trained model and all tables are included, so you can go straight to the dashboard:
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
To rebuild everything from the real data:
```bash
python -m src.data.real                 # real Ajman open data -> data/real/*.csv
python -m src.data.generate_synthetic   # firm-level records calibrated to real entries/exits
python -m src.models.km                 # survival curves + log-rank tests
python -m src.models.train              # Cox model, evaluation, dashboard tables
python -m src.forecast.forecast         # 2026-2028 license forecast + monthly free zone forecast
python -m src.forecast.hotspots         # sector and district growth outlook, free zone by license type
python -m src.growth.program            # 3 pillars: opportunity, marketing effects, creators, early warning, matching
pytest -q
```

Optional, for the live research agent:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.research.digital_agent "Business name" --area "Al Nuaimiya"
```

## Dashboard tabs
- **Entrepreneur**: score gauge, survival curve, risk drivers, "what if you post more?", best areas for your sector
- **Map**: interactive Folium map: score, sector & overall saturation, closures forecast, net new licenses, digital presence, market gaps, businesses; area popups with full DED profile
- **DED insights**: metric map with legend, saturation heatmap (3 views), area profiles, expected closures, average score by sector
- **Research a business**: Claude + web search estimates a real business's digital presence and feeds it into the score
- **Method**: the equations, learned weights, test metrics, calibration plot, fairness notes

## Results on the test cohort (licensed 2022+, synthetic)
| Metric | Value |
|---|---|
| C-index | 0.722 |
| AUC @ 3 yr: score / XGBoost | 0.749 / 0.737 |
| Brier @ 3 yr (model vs no-model) | 0.080 vs 0.098 |
| Hazard ratio: +1★ trusted rating | 0.37 |
| Hazard ratio: saturation 0 → 0.5 | 2.06 |

## Structure
```
src/data/       generate_synthetic.py, load.py
src/features/   saturation.py, build_features.py (Digital Presence Index, time split)
src/models/     km.py, cox.py, xgb_baseline.py, evaluate.py, score.py, train.py
src/research/   digital_agent.py (Claude API + web search)
src/viz/        risk_map.py (Folium maps)
app/            streamlit_app.py
data/geo/       ajman_areas.csv (approximate centroids + ESTIMATED population/visitors: replace with official data)
```

## Status
- [x] Step 1: synthetic data generator
- [x] Step 2: Kaplan-Meier curves + log-rank tests
- [x] Step 3: features, saturation, Digital Presence Index (learned weights)
- [x] Steps 4–7: Cox model, XGBoost baseline, λ fit, evaluation (C-index, IPCW Brier, calibration, PH check)
- [x] Step 8: score function, 12-month closure forecast, research agent, dashboard + interactive map
