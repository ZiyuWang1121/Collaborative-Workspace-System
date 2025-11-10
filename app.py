import os
import random
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from modeling import extract_skills_from_text, build_team_index, match_team_to_skills
from project_management import (
    initialize_project_store,
    create_project,
    add_task_to_project,
    update_task_status,
    update_task_progress,
    list_project_tasks,
    set_project_status,
    list_projects,
)
from analytics import render_dashboard


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEAM_PATH = os.path.join(DATA_DIR, "team_members.csv")
PROJECTS_PATH = os.path.join(DATA_DIR, "projects_sample.csv")
CSV_IN_ROOT = os.path.join(os.path.dirname(__file__), "JobsDatasetProcessed.csv")


def load_team_dataframe() -> pd.DataFrame:
    if os.path.exists(TEAM_PATH):
        return pd.read_csv(TEAM_PATH)
    # fallback: minimal synthetic
    return pd.DataFrame(
        [
            {"name": "Alex Johnson", "role": "Developer", "skills": "python,sql,ml"},
            {"name": "Jamie Smith", "role": "Data Scientist", "skills": "nlp,python,sklearn"},
            {"name": "Taylor Wu", "role": "Project Manager", "skills": "agile,planning,communication"},
        ]
    )


def maybe_load_reference_jobs() -> pd.DataFrame:
    if os.path.exists(CSV_IN_ROOT):
        try:
            # Only load a few columns/rows for light usage
            df = pd.read_csv(CSV_IN_ROOT, nrows=500)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def ensure_session():
    if "team_df" not in st.session_state:
        st.session_state.team_df = load_team_dataframe()
    if "jobs_df" not in st.session_state:
        st.session_state.jobs_df = maybe_load_reference_jobs()
    initialize_project_store()
    if "team_index" not in st.session_state:
        st.session_state.team_index = build_team_index(st.session_state.team_df)


def section_project_setup():
    st.header("Section 1: Project setup")
    with st.form("project_setup_form"):
        project_name = st.text_input("Project name")
        project_description = st.text_area("Project description", height=160)
        num_members = st.slider("Team size (auto-selected)", 1, 8, 4)
        submitted = st.form_submit_button("Create project")
    if submitted:
        if not project_name.strip():
            st.error("Please provide a project name.")
            return
        required_skills = extract_skills_from_text(project_description)
        matched = match_team_to_skills(
            team_index=st.session_state.team_index,
            required_skills=required_skills,
            k=num_members,
        )
        project_id = create_project(
            name=project_name,
            description=project_description,
            required_skills=required_skills,
            team_members=matched,
        )
        st.success(f"Project '{project_name}' created (ID: {project_id}).")
        st.subheader("Detected skills")
        st.write(", ".join(required_skills) if required_skills else "None")
        st.subheader("Selected team members")
        st.dataframe(pd.DataFrame(matched))


def section_project_tracking():
    st.header("Section 2: Project tracking")
    projects = list_projects()
    if not projects:
        st.info("No projects yet. Create one in Project setup.")
        return
    project_options = {f"{p['name']} (ID: {p['id']})": p["id"] for p in projects}
    project_label = st.selectbox("Select project", list(project_options.keys()))
    project_id = project_options[project_label]

    with st.expander("Add task", expanded=False):
        with st.form("add_task_form"):
            title = st.text_input("Task title")
            assignee = st.selectbox(
                "Assignee",
                [m["name"] for m in next(p for p in projects if p["id"] == project_id)["team_members"]],
            )
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            due_date = st.date_input("Due date")
            create_task = st.form_submit_button("Add task")
        if create_task:
            add_task_to_project(
                project_id=project_id,
                title=title,
                assignee=assignee,
                priority=priority,
                due_date=str(due_date),
            )
            st.success("Task added.")

    st.subheader("Tasks")
    tasks = list_project_tasks(project_id)
    if tasks:
        task_df = pd.DataFrame(tasks)
        st.dataframe(task_df)

        st.markdown("Update a task")
        with st.form("update_task_form"):
            if tasks:
                task_ids = [t["id"] for t in tasks]
                task_id = st.selectbox("Task ID", task_ids)
                new_status = st.selectbox("Status", ["Not Started", "In Progress", "Blocked", "Done"])
                new_progress = st.slider("Progress (%)", 0, 100, 0)
                do_update = st.form_submit_button("Update task")
            else:
                task_id = None
                do_update = False
            if do_update and task_id is not None:
                update_task_status(project_id, task_id, new_status)
                update_task_progress(project_id, task_id, int(new_progress))
                st.success("Task updated.")

    st.subheader("Project status")
    new_p_status = st.selectbox("Set project status", ["Planned", "Active", "On Hold", "Completed"])
    if st.button("Apply status"):
        set_project_status(project_id, new_p_status)
        st.success("Project status updated.")


def section_dashboard():
    st.header("Section 3: Dashboard")
    projects = list_projects()
    if not projects:
        st.info("No projects to analyze.")
        return
    render_dashboard(projects)


def main():
    st.set_page_config(page_title="Collaborative Workspace", layout="wide")
    st.title("Collaborative Workspace System")
    ensure_session()

    tab1, tab2, tab3 = st.tabs(["Project setup", "Project tracking", "Dashboard"])
    with tab1:
        section_project_setup()
    with tab2:
        section_project_tracking()
    with tab3:
        section_dashboard()


if __name__ == "__main__":
    main()


