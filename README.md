# Apollo Intelligence System

## Overview
An automated intelligence platform that ingests raw market competitive data and outputs an ExCo-ready dashboard with synthesized strategic insights.

## Prerequisites
- Ensure you have the `venv` activated.
- `ANTHROPIC_API_KEY` must be set in your environment variables.

## How to Run
1. Update `/knowledge/market_data.csv` with the latest competitive intelligence.
2. Open your terminal in the project root.
3. Run the pipeline:
   ```bash
   python agents.py