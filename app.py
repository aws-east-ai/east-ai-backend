from fastapi import FastAPI

# from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Router mapping
from controller.home import home_router
from controller.bedrock import bedrock_router
from controller.paint import paint_router
from controller.tools import tools_router
from controller.websocket import ws_router
from controller.agent import agent_router

app.include_router(home_router)
app.include_router(bedrock_router)
app.include_router(paint_router)
app.include_router(tools_router)
app.include_router(ws_router)
app.include_router(agent_router)
