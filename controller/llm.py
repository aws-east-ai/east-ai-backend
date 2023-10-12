from fastapi import APIRouter


# 解析 stream
class StreamScanner:
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0

    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    def readlines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line[-1] != b"\n":
                self.read_pos += len(line)
                yield line[:-1]

    def reset(self):
        self.read_pos = 0


class LLM:
    def __init__(self):
        self.router = APIRouter()
        # self.router.add_api_route("/", self.home, methods=["GET"])

    # async def home(self):
    #     return {"message": "Welcome to the GenAI workshop"}


llm_router = LLM().router
