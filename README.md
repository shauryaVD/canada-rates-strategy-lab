# Canada Rates Strategy Lab

Canada Rates Strategy Lab is a Python research application for monitoring Government of Canada benchmark bond yields, studying curve dynamics, screening term spreads for statistical deviations, and evaluating historical mean-reversion hypotheses. It turns official daily Bank of Canada observations into transparent research outputs without presenting the analysis as a trading terminal or an investment recommendation.

The project was built as a public fixed-income research portfolio for interest-rate strategy and quantitative research roles. It emphasizes reproducible calculations, careful data lineage, explicit missing-data handling, and a backtest that reports basis-point outcomes instead of fictional investment returns.

**[Open the live research dashboard](https://canada-rates-strategy-lab.onrender.com/)**

## Application preview

The Streamlit interface uses a restrained navy research-workstation theme with five pages: Market Overview, Yield Curve, Relative Value, Backtesting, and Research Brief. The public Render deployment and all five pages have been checked at desktop width with live Bank of Canada observations.

## Core capabilities

- Monitor official 2-, 3-, 5-, 7-, and 10-year Government of Canada benchmark yields.
- Compare current, prior, five-observation, and twenty-observation yield movements.
- Plot historical yields, benchmark curve points, and 2s10s, 2s5s, and 5s10s spreads.
- Classify bull/bear steepening and flattening only when endpoint moves support the label.
- Calculate prior-window rolling means, sample standard deviations, z-scores, and midrank percentiles.
- Evaluate high- and low-z-score events over 5-, 10-, or 20-observation horizons.
- Audit excluded events, missing paths, train/test boundary crossings, and overlapping windows.
- Compare threshold events with an unconditional historical baseline.
- Export observations, signals, outcomes, summaries, assumptions, Markdown, and a one-page PDF brief.

## Technology stack

- Python 3.11+
- pandas and NumPy
- Streamlit
- Plotly
- requests
- ReportLab
- pytest and Ruff
- GitHub Actions

## Data source and coverage

Data comes from the official [Bank of Canada Valet API](https://www.bankofcanada.ca/valet-api-how-to/) and the Bank's [selected bond yield publication](https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/). No registration or API key is required.

The application requests these published benchmark series:

| Maturity | Valet series |
| --- | --- |
| 2-year | `BD.CDN.2YR.DQ.YLD` |
| 3-year | `BD.CDN.3YR.DQ.YLD` |
| 5-year | `BD.CDN.5YR.DQ.YLD` |
| 7-year | `BD.CDN.7YR.DQ.YLD` |
| 10-year | `BD.CDN.10YR.DQ.YLD` |

The live integration check on October 31, 2026 returned 6,458 observation dates from January 2, 2001 through October 29, 2026. Actual coverage can change as the source updates. The application displays the current coverage, fetch timestamp, latest observation date, cache status, and validation warnings.

Requests use bounded timeouts, retries, and exponential backoff. A successful response is validated before it is saved to an atomic, checksummed six-hour cache. If a refresh fails, a previously validated real-data cache may be used and is labeled `stale cache`. The pipeline does not forward-fill missing yields, invent holiday rows, or substitute another series.

## Installation and local use

```bash
git clone https://github.com/shauryaVD/canada-rates-strategy-lab.git
cd canada-rates-strategy-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. Use the sidebar to select the observation window, yield maturities, curve spread, prior reference window, z-score threshold, forward horizon, evaluation sample, and overlap rule.

To generate the same real-data brief outside Streamlit:

```bash
python -m scripts.export_sample_brief
```

The command writes a PDF under `output/pdf/` and Markdown under `output/markdown/`. These local outputs are ignored by Git because their values and dates change with the data source.

## Example analytical workflow

1. Review the latest 2-, 5-, and 10-year yields and their one-, five-, and twenty-observation changes.
2. Compare the current benchmark curve with five and twenty observations earlier or a selected historical date.
3. Select a term spread and evaluate its z-score against 20, 60, or 120 prior observations.
4. Test the historical hypothesis that an unusually wide spread subsequently narrows, or an unusually narrow spread subsequently widens.
5. Review signal exclusions, event outcomes, horizon sensitivity, and the unconditional baseline.
6. Export the underlying data and deterministic research brief.

Using the default three-year sample on October 29, 2026, the 2s10s spread was 48 bps. Its z-score against the preceding 60 observations was -2.21. One real historical high-spread event occurred on August 17, 2026 with a spread of 75 bps and a z-score of +1.89; ten observations later, on August 31, the spread was 72 bps, a -3 bps move that supported the narrowing hypothesis for that event. Values are descriptive observations from the cited public series and will change as the selected sample or parameters change.

## Methodology

Term spreads are measured in basis points:

```text
2s10s = (10-year yield - 2-year yield) × 100
2s5s  = (5-year yield - 2-year yield) × 100
5s10s = (10-year yield - 5-year yield) × 100
```

Yields are supplied in percentage points, so one percentage point equals 100 basis points.

For spread observation `S(t)`, the rolling z-score is:

```text
z(t) = [S(t) - mean(S(t-window), ..., S(t-1))]
       / std(S(t-window), ..., S(t-1))
```

The reference window excludes the current observation and uses the sample standard deviation (`ddof=1`). A complete prior window is required. Zero variance, missing values, and insufficient history produce no z-score.

## Backtesting assumptions

The backtester is an event study of future spread changes, not a bond portfolio simulator.

- Signals use only completed data available at the signal timestamp.
- A chronological 70/30 split defines the training and test samples.
- Training events whose endpoints cross the split are purged.
- Events without the full future path are excluded.
- Missing observations anywhere from signal through endpoint exclude that event.
- Optional non-overlap uses a documented greedy chronological rule.
- The unconditional baseline retains overlapping complete paths and is contextual rather than a matched independent control.
- A hit means strictly positive direction-adjusted convergence; zero is a miss.
- Parameters are user-selected. Reviewing and tuning them against test results makes the exercise exploratory rather than an untouched out-of-sample claim.

The exported audit trail preserves valid signals, exclusion reasons, evaluated outcomes, split date, overlap rule, missing-path rule, and all selected parameters.

## Important distinctions and limitations

- A yield change is not a bond return. Duration, convexity, carry, and cash flows are not modeled.
- A statistical extreme is not proof of economic mispricing. Policy expectations, inflation, term premiums, liquidity, and regime shifts require separate evidence.
- Historical spread convergence is not a tradable profit. A strategy would require DV01-balanced exposures, curve hedges, roll-down, execution, transaction costs, and financing.
- The plotted maturities are selected benchmark bond yields connected for comparison. They are not a bootstrapped or fitted zero-coupon curve.
- The source-supplied observation grid is authoritative. Without a separate market calendar, absent rows cannot be independently classified as holidays or publication gaps.
- Consecutive threshold dates may describe one persistent episode. Overlap and parameter selection limit independence and generalization.
- No macroeconomic catalyst is inferred from yield data alone.

## Tests and quality checks

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
```

The test suite covers API parsing and metadata, invalid and missing values, duplicates, cache integrity and fallback behavior, basis-point conversions, curve classifications, rolling statistics, chronological splits, future perturbation and prefix invariance, missing forward paths, overlapping events, report values, and one-page PDF generation. GitHub Actions runs lint and tests on Python 3.11 and 3.13 for pushes and pull requests.

## Deploy on Render

The included `render.yaml` deploys the application as one free Render Python web service, following the same portfolio-hosting pattern as CreditPath. It installs `requirements.txt`, starts Streamlit on Render's assigned `PORT`, and checks `/_stcore/health`. No secrets, paid APIs, databases, or environment variables are required.

Public URL: **https://canada-rates-strategy-lab.onrender.com/**

The free service can spin down while idle, so the first recruiter visit after inactivity may take longer to load. Subsequent navigation is interactive once the service is awake.

## Research-only disclaimer

This independent personal project is not affiliated with RBC, the Bank of Canada, or any financial institution. It is provided for reproducible research and educational purposes only. It is not investment advice, a trade recommendation, or evidence of expected profitability.
