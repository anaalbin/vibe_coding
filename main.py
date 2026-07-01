import asyncio

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    desired_role: str
    top_n: int = 20


class JobFit(BaseModel):
    job_index: int
    score: int
    reason: str


class JobFitRanking(BaseModel):
    ranked_jobs: list[JobFit]


anthropic_client = anthropic.Anthropic()


@app.post("/rank-jobs")
async def rank_jobs(payload: RankJobsRequest) -> dict:
    """Rank open jobs by how well they fit a candidate's CV and desired role."""
    jobs = await fetch_all_jobs()
    if not jobs:
        raise HTTPException(status_code=502, detail="No jobs available to rank")

    job_listing = "\n".join(
        f"{i}. {job['title']} at {job['company']} ({job['location'] or 'location unspecified'})"
        for i, job in enumerate(jobs)
    )

    response = anthropic_client.messages.create(
        model="claude-opus-4-8",
        max_tokens=8000,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked_jobs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "job_index": {"type": "integer"},
                                    "score": {"type": "integer"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["job_index", "score", "reason"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["ranked_jobs"],
                    "additionalProperties": False,
                },
            }
        },
        messages=[
            {
                "role": "user",
                "content": (
                    "A candidate is looking for a role as: "
                    f"{payload.desired_role}\n\n"
                    f"Candidate's CV:\n{payload.cv}\n\n"
                    f"Open jobs (numbered):\n{job_listing}\n\n"
                    f"Pick the {payload.top_n} jobs from the list above that best fit this "
                    "candidate's CV and desired role. For each, give a fit score from 0-100 "
                    "(100 = perfect fit) and a short one-sentence reason referencing specifics "
                    "from the CV and the job. Order the results from best to worst fit."
                ),
            }
        ],
    )

    text = next(block.text for block in response.content if block.type == "text")
    ranking = JobFitRanking.model_validate_json(text)

    ranked_jobs = []
    for fit in ranking.ranked_jobs:
        if 0 <= fit.job_index < len(jobs):
            ranked_jobs.append({**jobs[fit.job_index], "score": fit.score, "reason": fit.reason})
    ranked_jobs.sort(key=lambda job: job["score"], reverse=True)

    return {
        "desired_role": payload.desired_role,
        "count": len(ranked_jobs),
        "ranked_jobs": ranked_jobs,
    }
