# mlops-labs

Predictive alerts for abnormal S&P 500 moves around Trump's Truth Social posts (AI290 MLOps, KSE).

## Problem

Posts by Donald Trump on Truth Social appear at any time of day and are sometimes followed by sharp moves in the stock market. This project builds a service that, for every new post published during US market hours, estimates the probability that the S&P 500 makes an abnormally large move over the next two hourly bars. A move is abnormal when its absolute log return exceeds 1.5 standard deviations of the same time-slot return over the previous 20 trading days, so the task is binary classification. The service raises an alert for the top 20% of posts by predicted probability, so that an analyst looks at the market sooner. It is a predictive tool only and makes no causal claim that a post causes a market move.

Full design: [docs/design.md](docs/design.md).

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | `docs/design.md` follows the template, every section filled or marked as an assumption | Lab 1 |

## Setup

```bash
make setup   # uv sync: creates .venv and installs pinned dependencies
make lint    # ruff check
```

Without `make` (e.g. Windows):

```powershell
uv sync
uv run ruff check .
```

## Data

- Posts: [chrissoria/trump-truth-social](https://huggingface.co/datasets/chrissoria/trump-truth-social) on Hugging Face (CC-BY-NC-4.0).
- Prices: hourly `^GSPC` bars from Yahoo Finance via `yfinance`.

`data/raw/` is git-ignored. Data versioning with DVC arrives in Lab 2.

## Use of AI tools

I used Claude as an assistant for this lab

1. Topic discussion and ideation
2. Data research and analysis
3. Drafting the design document, architecture diagrams and README
4. Architecture planning

This part is written by me, by hand. Everything is reviewed, verified and approved by me.