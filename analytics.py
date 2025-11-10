from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd
import altair as alt
import streamlit as st


def _tasks_frame(projects: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in projects:
        for t in p.get("tasks", []):
            rows.append(
                {
                    "project": p["name"],
                    "task_id": t["id"],
                    "title": t["title"],
                    "assignee": t["assignee"],
                    "priority": t["priority"],
                    "status": t["status"],
                    "progress": t["progress"],
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["project", "task_id", "title", "assignee", "priority", "status", "progress"])


def _team_frame(projects: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in projects:
        for m in p.get("team_members", []):
            rows.append(
                {
                    "project": p["name"],
                    "name": m.get("name", ""),
                    "role": m.get("role", ""),
                    "skills": m.get("skills", ""),
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["project", "name", "role", "skills"])


def render_dashboard(projects: List[Dict[str, Any]]) -> None:
    tasks_df = _tasks_frame(projects)
    team_df = _team_frame(projects)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project progress")
        if not tasks_df.empty:
            proj_progress = tasks_df.groupby("project")["progress"].mean().reset_index()
            chart = (
                alt.Chart(proj_progress)
                .mark_bar()
                .encode(x="project:N", y=alt.Y("progress:Q", title="Avg Progress (%)"))
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No tasks yet.")

        st.subheader("Tasks by status")
        if not tasks_df.empty:
            status_count = tasks_df.groupby(["project", "status"]).size().reset_index(name="count")
            chart2 = (
                alt.Chart(status_count)
                .mark_bar()
                .encode(x="project:N", y="count:Q", color="status:N", column="status:N")
                .properties(height=300)
            )
            st.altair_chart(chart2, use_container_width=True)
        else:
            st.info("No task status data.")

    with col2:
        st.subheader("Skill set distribution")
        if not team_df.empty:
            # explode skills
            exploded = team_df.assign(skill=team_df["skills"].str.split(",")).explode("skill")
            exploded["skill"] = exploded["skill"].fillna("").str.strip().replace("", pd.NA)
            exploded = exploded.dropna(subset=["skill"])
            if not exploded.empty:
                skill_counts = exploded.groupby(["project", "skill"]).size().reset_index(name="count")
                chart3 = (
                    alt.Chart(skill_counts)
                    .mark_bar()
                    .encode(x="skill:N", y="count:Q", color="project:N")
                    .properties(height=300)
                )
                st.altair_chart(chart3, use_container_width=True)
            else:
                st.info("No skills listed for team members.")
        else:
            st.info("No team members yet.")

        st.subheader("Role allocation overview")
        if not team_df.empty:
            role_counts = team_df.groupby(["project", "role"]).size().reset_index(name="count")
            chart4 = (
                alt.Chart(role_counts)
                .mark_bar()
                .encode(x="role:N", y="count:Q", color="project:N")
                .properties(height=300)
            )
            st.altair_chart(chart4, use_container_width=True)
        else:
            st.info("No roles to display.")


