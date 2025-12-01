# main.py
import streamlit as st
from app import ProjectManagementSystem
from model import predict_budget,select_optimal_team,get_skills_from_description

pms = ProjectManagementSystem()

# Set external models
pms.set_recommendation_model(select_optimal_team)
pms.set_task_recommendation_model(predict_budget)

# Run the application
pms.run()