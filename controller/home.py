from fastapi import APIRouter


class Home:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/", self.home, methods=["GET"])

    async def home(self):
        return {"message": "Welcome to the GenAI workshop"}


home_router = Home().router
