# import os
from fastapi import APIRouter
from utils.painter import product_design, inpaint


class Paint:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route(
            "/api/product-design", self.route_pd_design, methods=["POST"]
        )
        self.router.add_api_route("/api/inpaint", self.route_inpaint, methods=["POST"])

    def route_pd_design(self, item: dict):
        return product_design(item)

    def route_inpaint(self, item: dict):
        return inpaint(item)


paint_router = Paint().router
