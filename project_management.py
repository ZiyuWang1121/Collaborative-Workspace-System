from __future__ import annotations

import uuid
from typing import Dict, Any, List

import streamlit as st


def _ensure_store():
    if "projects_store" not in st.session_state:
        st.session_state.projects_store = {}


def initialize_project_store():
    _ensure_store()


def create_project(name: str, description: str, required_skills: List[str], team_members: List[Dict[str, Any]]) -> str:
    _ensure_store()
    pid = str(uuid.uuid4())[:8]
    st.session_state.projects_store[pid] = {
        "id": pid,
        "name": name,
        "description": description,
        "required_skills": required_skills,
        "team_members": team_members,
        "status": "Planned",
        "tasks": [],
    }
    return pid


def list_projects() -> List[Dict[str, Any]]:
    _ensure_store()
    return list(st.session_state.projects_store.values())


def add_task_to_project(
    project_id: str,
    title: str,
    assignee: str,
    priority: str,
    due_date: str,
) -> str:
    _ensure_store()
    tid = str(uuid.uuid4())[:8]
    task = {
        "id": tid,
        "title": title,
        "assignee": assignee,
        "priority": priority,
        "due_date": due_date,
        "status": "Not Started",
        "progress": 0,
    }
    st.session_state.projects_store[project_id]["tasks"].append(task)
    return tid


def list_project_tasks(project_id: str) -> List[Dict[str, Any]]:
    _ensure_store()
    return list(st.session_state.projects_store.get(project_id, {}).get("tasks", []))


def update_task_status(project_id: str, task_id: str, status: str) -> None:
    _ensure_store()
    tasks = st.session_state.projects_store[project_id]["tasks"]
    for t in tasks:
        if t["id"] == task_id:
            t["status"] = status
            return


def update_task_progress(project_id: str, task_id: str, progress: int) -> None:
    _ensure_store()
    tasks = st.session_state.projects_store[project_id]["tasks"]
    for t in tasks:
        if t["id"] == task_id:
            t["progress"] = max(0, min(100, int(progress)))
            return


def set_project_status(project_id: str, status: str) -> None:
    _ensure_store()
    st.session_state.projects_store[project_id]["status"] = status


