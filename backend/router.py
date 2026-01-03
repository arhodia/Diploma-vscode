# algorithms/router.py
from typing import Tuple

def run_pipeline(selected_option, file_type, algorithm):
    algo = (algorithm or "").upper().strip()

    if algo == "KMEANS":
        from backend.algorithms.kmeans import run_kmeans as runner
        return runner(selected_option, file_type)

    if algo == "BISECTING_KMEANS":
        from backend.algorithms.bisc_kmeans import run as runner
        return runner(selected_option, file_type)

    if algo == "DBSCAN":
        from backend.algorithms.dbscan import run as runner
        return runner(selected_option, file_type)

    if algo == "LSH":
        from backend.algorithms.lsh import run as runner
        return runner(selected_option, file_type)

    raise ValueError(f"Unknown algorithm: {algorithm}")