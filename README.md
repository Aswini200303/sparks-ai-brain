# Sparks AI Brain

Sparks AI Brain is a FastAPI microservice that analyzes Amazon Ads search term performance with Google Gemini AI and returns structured campaign optimization actions.

## What it does

- Accepts search term metrics for Amazon Ads
- Uses Google Gemini to assess performance
- Returns recommended actions like:
  - `increase_bid`
  - `decrease_bid`
  - `add_negative`
  - `no_action`

## Why it exists

This service is designed to support automated campaign optimization workflows by converting search term performance data into actionable recommendations, especially for nightly optimization pipelines.

## Quick start

1. Clone the repository
2. Copy `.env.example` to `.env`
3. Fill in `GEMINI_API_KEY`, `SERVICE_API_KEY`, and other settings
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
Run the app:
API
GET /health — health check
POST /api/agent/harvest — analyze search terms and get optimization actions
Environment

Required:
GEMINI_API_KEY
SERVICE_API_KEY

Optional:
HOST (0.0.0.0)
PORT (8000)
DEBUG
GEMINI_MODEL
GEMINI_TEMPERATURE
   
