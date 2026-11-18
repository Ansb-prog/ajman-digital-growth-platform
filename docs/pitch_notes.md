# Pitch notes: Ajman Digital Growth Platform

## How to say it (claims that survive a judge's questions)
- Say: "A forecasting engine on real Ajman data, and a scoring method demonstrated on a simulation calibrated to DED's real yearly totals, ready to retrain on DED's license history."
- Don't say: "Our model found that one star cuts risk by 63%" or "73% accuracy". Those numbers come from the simulation's assumptions.
- Say: "The simulation reproduces DED's yearly exits and active licenses." Don't say "validated within 0.5%".
- Say: "The marketing effect is an assumption (−20% closure risk, range 10–30%) that the pilot will measure."
- Lead with the forecast engine (the challenge brief). The three pillars show how DED and FZA use it.

## Slide: the equation (lead slide after the problem)
Active(t+1) = Active(t) + New(t+1) − Exits(t+1)
Exits(t+1) = Σ_i h_i(age, sector, district, reputation, digital presence) × L(t+1)
- New licenses: trend + damped Holt. Backtest error 1.0% (naive 19.8%).
- Active licenses: backtest error 1.8% (naive 1.9%).
- Exits: backtest error 25% (naive 10%). This is the weakest part; say so.
- 2026: about 45,800 active licenses, 8,200 new, 3,700 exits (ranges in the app).

## Slide: cost per outcome (default assumptions, adjustable in the app)
Voucher AED 1,250 per business per month (50% of a creator's fee), 6 months, +15% admin.

| Scenario, 2026–2028 | Businesses matched | Cost (AED m) | Closures avoided | AED per closure avoided |
|---|---|---|---|---|
| Pilot, 200 a year | 600 | 5.2 | 38 (19–59) | ~135k (88k–275k) |
| Program, 1,000 a year | 3,000 | 25.9 | 192 (94–294) | ~135k (88k–275k) |

Not counted: extra revenue of matched businesses, new creator licenses (FZA revenue), and renewal fees kept.
Message: "Expensive if it only saves closures. The pilot tells us if it also grows revenue and creator licensing. That's why we start with 200, not 3,000."

## Slide: limitations (own them before the judges find them)
1. No public data follows individual businesses, so firm-level records are simulated and calibrated to real totals.
2. The marketing effect size is assumed, and the pilot measures it.
3. The exit forecast is the weakest part (backtest).
4. Residents per district are estimates.
5. District growth is affected by data gaps (Apr–Dec 2024), renamed districts and Masfout/Manama registrations.
6. Sectors come from keyword rules on DED activity names (98% mapped).

## Slide: the ask
- DED: anonymized license history, to retrain on real outcomes.
- FZA: freelance and media licenses by activity (AFZ + Ajman Media City).
- Statistics Centre: residents by district.
- A 200 + 200 business pilot for 12 months.

## Fairness and privacy (one line each, say it out loud)
- The registration score is advisory, never a licensing condition. It uses no nationality, gender or age, and no guessed marketing data.
- The early-warning list is used only for opt-in support offers, under the UAE data protection law (PDPL). There's no public ranking of businesses.
- Suggestions are capped: a district is never suggested above saturation 0.30, and suggestions rotate within each district's capacity.

## Demo (3 screens, about 3 minutes, with a backup video)
1. Forecast engine: the equation, the 2026–2028 forecast, and sector/district hotspots.
2. Registration Advisor: F&B in Nuaimeya 1, then the suggestions (Nakheel 2 / Al Helio 2 / Al Zahya), with capacity and incentive.
3. Growth Matchmaker: sector effects (assumption badge), one business card, the cost table and the pilot.
