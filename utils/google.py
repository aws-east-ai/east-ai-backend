import os
from googleapiclient.discovery import build

google_api_key = os.environ.get("GOOGLE_API_KEY")
goole_cse_cx = os.environ.get("GOOGLE_CSE_CX")


def get_cse_result(q: str):
    service = build("customsearch", "v1", developerKey=google_api_key)
    res = service.cse().list(q=q, cx=goole_cse_cx).execute()
    return res["items"]
