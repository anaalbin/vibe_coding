import asyncio
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="Reeds Jobs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREENHOUSE_BOARDS = [
    "riskified",
    "fireblocks",
    "pagayais",
    "gongio",
    "lightricks",
    "similarweb",
    "melio",
    "wizinc",
    "yotpo",
    "catonetworks",
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    """Fetch all jobs for a single Greenhouse board and tag them with the company."""
    response = await client.get(GREENHOUSE_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("name"),
                "apply_url": job.get("absolute_url"),
                "company": token,
            }
        )
    return jobs


async def fetch_all_jobs() -> list[dict]:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_board(client, token) for token in GREENHOUSE_BOARDS),
            return_exceptions=True,
        )

    jobs: list[dict] = []
    for token, result in zip(GREENHOUSE_BOARDS, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for board '{token}': {result}",
            )
        jobs.extend(result)

    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    jobs = await fetch_all_jobs()
    return {"count": len(jobs), "jobs": jobs}


class RankJobsRequest(BaseModel):
    cv: str
    role: str
    top_n: int = 20


class JobFit(BaseModel):
    job_index: int
    score: int
    reason: str


class JobFitRanking(BaseModel):
    ranked_jobs: list[JobFit]


GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


@app.post("/rank")
async def rank_jobs(payload: RankJobsRequest) -> dict:
    """Rank open jobs by how well they fit a candidate's CV and desired role."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
    gemini_client = genai.Client(api_key=api_key)

    jobs = await fetch_all_jobs()
    if not jobs:
        raise HTTPException(status_code=502, detail="No jobs available to rank")

    job_listing = "\n".join(
        f"{i}. {job['title']} at {job['company']} ({job['location'] or 'location unspecified'})"
        for i, job in enumerate(jobs)
    )

    prompt = (
        f"A candidate is looking for a role as: {payload.role}\n\n"
        f"Candidate's CV:\n{payload.cv}\n\n"
        f"Open jobs (numbered):\n{job_listing}\n\n"
        f"Pick the {payload.top_n} jobs from the list above that best fit this candidate's "
        "CV and desired role. For each, give a fit score from 0-100 (100 = perfect fit) and "
        "a short one-sentence reason referencing specifics from the CV and the job. Order "
        "the results from best to worst fit."
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobFitRanking,
        ),
    )
    ranking = JobFitRanking.model_validate_json(response.text)

    ranked_jobs = []
    for fit in ranking.ranked_jobs:
        if 0 <= fit.job_index < len(jobs):
            ranked_jobs.append({**jobs[fit.job_index], "score": fit.score, "reason": fit.reason})
    ranked_jobs.sort(key=lambda job: job["score"], reverse=True)

    return {
        "role": payload.role,
        "count": len(ranked_jobs),
        "ranked_jobs": ranked_jobs,
    }
