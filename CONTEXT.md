# Project Context: Apollo Intelligence System

## Objective
To build a high-fidelity, automated Competitive Intelligence Engine for Apollo Africa. The system replaces manual, Excel-based reporting with an automated, AI-synthesized intelligence dashboard designed for ExCo-level decision-making.

## Architecture
- **Input**: CSV data in `/knowledge/market_data.csv`.
- **Processing**: Python pipeline (`agents.py`) using `pandas` for data structuring, `tenacity` for API resilience, and `anthropic` for strategic analysis.
- **Output**: An interactive `export/dashboard.html` that visualizes competitive data and provides actionable strategic memos.

## Strategic Framework
The system should evaluate competitors based on:
1. **Market Velocity Index**: (Deals Logged / Time in Market).
2. **Regulatory Fragility Score**: Comparative analysis of sentiment from regulatory updates vs. municipal footprint.
3. **Strategic Strike**: Identifies high-potential municipal nodes for aggressive contracting.

## Current Status (as of April 2026)
- The technical "plumbing" (API authentication, environment, library management) is complete and functional.
- The pipeline is resilient to transient API failures (using tenacity).
- We are transitioning from simple bar chart generation to an interactive KPI dashboard.