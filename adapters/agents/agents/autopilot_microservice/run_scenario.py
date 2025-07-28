import os
import subprocess
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ScenarioRequest(BaseModel):
    route_id: int
    repetitions: int = 1


@router.post("/run_scenario/")
def run_scenario(req: ScenarioRequest):
    current_file = os.path.abspath(__file__)
    project_root = current_file.split("MitsuNeuroPilotAPIRESTRUCTURED")[0] + "MitsuNeuroPilotAPIRESTRUCTURED"

    leaderboard_root = os.path.join(project_root, "services", "evaluation_service", "leaderboard", "leaderboard")

    # ПРАВИЛЬНЫЙ ПУТЬ к merged_routes.xml
    routes_path = os.path.join(project_root, "services", "evaluation_service", "leaderboard", "data",
                               "merged_routes.xml")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["LEADERBOARD_ROOT"] = leaderboard_root
    os.environ["PYTHONPATH"] = project_root
    os.environ["TEAM_AGENT"] = "adapters.agents.agents.autopilot_service.agent"
    os.environ["ROUTES"] = routes_path
    os.environ["ROUTES_SUBSET"] = str(req.route_id)
    os.environ["REPETITIONS"] = str(req.repetitions)
    os.environ["DEBUG_CHALLENGE"] = "0"
    os.environ["CHALLENGE_TRACK_CODENAME"] = "SENSORS"
    os.environ["CHECKPOINT_ENDPOINT"] = os.path.join(project_root, "services", "evaluation_service", "data",
                                                     "autopilot_behavior_data", "results2.json")
    os.environ["RECORD_PATH"] = ""
    os.environ["RESUME"] = ""

    command = [
        "python", os.path.join(leaderboard_root, "leaderboard_evaluator.py"),
        f"--routes={os.environ['ROUTES']}",
        f"--track={os.environ['CHALLENGE_TRACK_CODENAME']}",
        f"--checkpoint={os.environ['CHECKPOINT_ENDPOINT']}",
        f"--agent={os.environ['TEAM_AGENT']}",
        f"--agent-config=",  # опционально
        f"--debug={os.environ['DEBUG_CHALLENGE']}",
        f"--resume={os.environ['RESUME']}"
    ]

    route_exists = os.path.exists(routes_path)

    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
        return {
            "status": "ok",
            "route_file": routes_path,
            "route_file_exists": route_exists,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "route_file": routes_path,
            "route_file_exists": route_exists,
            "stdout": e.stdout,
            "stderr": e.stderr
        }
