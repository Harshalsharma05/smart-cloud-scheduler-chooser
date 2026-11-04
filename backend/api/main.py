"""FastAPI app: GET /health, /scenarios, /rulesets, /profiles, POST /run. CORS open for the Vite dev server.
Run from backend/:  uvicorn api.main:app --reload"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eval.runner import run_comparison
from rules.library import RULESETS
from sim.models import load_config
from workloads.scenarios import SCENARIOS

CFG = load_config()
PROFILES = CFG["score"]["profiles"]

app = FastAPI(title="Smart Scheduler Chooser")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class RunRequest(BaseModel):
    scenario: str = "rush"
    seed: int = 1
    rulesets: list[str] = list(RULESETS)
    timeline_for: str = "B4"
    profile: str = CFG["score"]["default_profile"]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/scenarios")
def scenarios():
    return [{"id": k, "label": v["label"], "description": v["description"]} for k, v in SCENARIOS.items()]


@app.get("/rulesets")
def rulesets():
    return [{"id": r.id, "order": r.order, "backfill": r.backfill, "burst": r.burst, "label": r.label}
            for r in RULESETS.values()]


@app.get("/profiles")
def profiles():
    return [{"id": k, **w} for k, w in PROFILES.items()]


@app.post("/run")
def run(req: RunRequest):
    if req.scenario not in SCENARIOS:
        raise HTTPException(400, f"unknown scenario {req.scenario}")
    bad = [r for r in req.rulesets + [req.timeline_for] if r not in RULESETS]
    if bad or not req.rulesets:
        raise HTTPException(400, f"unknown or empty rule sets: {bad}")
    if req.profile not in PROFILES:
        raise HTTPException(400, f"unknown profile {req.profile}")
    return run_comparison(req.scenario, req.seed, req.rulesets, req.timeline_for, req.profile, CFG)
