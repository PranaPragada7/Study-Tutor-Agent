"""Local FastAPI surface for Study Tutor profiles and progress summaries."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from services.runtime import get_runtime_config
from utils.student_profile import (
    create_profile,
    delete_profile,
    get_performance_summary,
    list_saved_profiles,
    load_profile,
)


class CourseCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=2, max_length=160)
    difficulty: int = Field(default=3, ge=1, le=5)
    exam_date: str = Field(default="", max_length=32)


class ProfileCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    learning_style: str = Field(default="balanced", pattern="^(balanced|concise|detailed|visual)$")
    courses: list[CourseCreate] = Field(min_length=1, max_length=12)


app = FastAPI(
    title="Study Tutor Local API",
    version="1.1.0",
    description=(
        "Local-only API for profile setup and learning-progress inspection. "
        "Bind to 127.0.0.1; this project does not provide shared-service authentication."
    ),
)


@app.get("/health")
def health() -> dict[str, Any]:
    runtime = get_runtime_config()
    return {
        "status": "ok",
        "ai_mode": runtime.mode,
        "storage": runtime.storage_backend,
        "scope": "local-only",
    }


@app.get("/api/v1/profiles")
def profiles() -> list[dict]:
    return list_saved_profiles()


@app.post("/api/v1/profiles", status_code=status.HTTP_201_CREATED)
def create(payload: ProfileCreate) -> dict:
    if load_profile(payload.name) is not None:
        raise HTTPException(status_code=409, detail="A profile with this name already exists.")
    return create_profile(
        payload.name,
        [course.model_dump() for course in payload.courses],
        learning_style=payload.learning_style,
    )


@app.get("/api/v1/profiles/{student_name}")
def profile(student_name: str) -> dict:
    loaded = load_profile(student_name)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return loaded


@app.get("/api/v1/profiles/{student_name}/summary")
def summary(student_name: str) -> dict:
    loaded = load_profile(student_name)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return get_performance_summary(loaded)


@app.delete("/api/v1/profiles/{student_name}", status_code=status.HTTP_204_NO_CONTENT)
def remove(student_name: str) -> None:
    if load_profile(student_name) is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    if not delete_profile(student_name):
        raise HTTPException(status_code=503, detail="Profile could not be deleted.")
