from fastapi import APIRouter

router = APIRouter(prefix="/passnet", tags=["passnet"])

@router.get("/health")
def passnet_health():
    return {"ok": True, "repo": "runpod-football-analyzer"}

@router.get("/{job_id}")
def passnet_stub(job_id: str):
    # static demo graph; just to validate shape
    return {
        "job_id": job_id,
        "teams": [
            {
                "team_id": 0,
                "nodes": [
                    {"id": 11, "label": "11", "x": 10, "y": 30},
                    {"id": 7,  "label": "7",  "x": 40, "y": 50},
                    {"id": 9,  "label": "9",  "x": 70, "y": 40}
                ],
                "links": [
                    {"source": 11, "target": 7, "weight": 3},
                    {"source": 7,  "target": 9, "weight": 2}
                ]
            },
            {
                "team_id": 1,
                "nodes": [
                    {"id": 4, "label": "4", "x": 20, "y": 60},
                    {"id": 6, "label": "6", "x": 50, "y": 30}
                ],
                "links": [
                    {"source": 4, "target": 6, "weight": 1}
                ]
            }
        ]
    }