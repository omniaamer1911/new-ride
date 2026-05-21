import os
from dotenv import load_dotenv

load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

from app.api.chat import router as chat_router
from app.api.deps import get_geocoder, get_llm, get_session
from app.api.drivers import router as drivers_router
from app.api.trips import router as trips_router
from app.chatbot.engine import run_chat_turn
from app.chatbot.providers.base import LLMProvider
from app.db.seed import seed_if_empty
from app.db.session import async_session_factory, init_db, dispose_engine
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.geocoding import LandmarkGeocoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. الـ Geocoder هيشتغل في الحالتين (لوكال وفيرسل) بدون مشاكل
    geocoder = LandmarkGeocoder()
    app.state.geocoder = geocoder

    # 2. الشرط الذكي: إنشاء الجداول وحقن البيانات (Seed) هيشتغل لوكال بس
    # وعلى فيرسل هيتخطاه تماماً عشان السيرفر ما يضربش كراش وقت الـ Startup
    if not os.getenv("VERCEL"):
        await init_db()
        
        factory = async_session_factory()
        async with factory() as session:
            await seed_if_empty(session)
            await session.commit()

    yield
    # تأمين قفل الـ engine عند إيقاف السيرفر
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(title="Ride Chatbot Egypt", lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(trips_router)
    app.include_router(drivers_router)

    @app.get("/")
    async def root():
        return {
            "ok": True,
            "service": "Ride Chatbot Egypt",
            "endpoints": {
                "chat": "POST /chat",
                "chat_api": "POST /api/chat",
                "health": "GET /healthz",
            },
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat_simple(
        body: ChatRequest,
        session: AsyncSession = Depends(get_session),
        geocoder=Depends(get_geocoder),
        llm: LLMProvider = Depends(get_llm),
    ) -> ChatResponse:
        reply = await run_chat_turn(
            session,
            geocoder,
            llm,
            body.user_id,
            body.message,
        )
        return ChatResponse(reply_ar=reply)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


app = create_app()