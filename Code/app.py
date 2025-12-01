# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import uuid
import io
import plotly.express as px
import plotly.graph_objects as go

class ProjectManagementSystem:
    """Project Management System Class"""
    
    def __init__(self):
        # Initialize session state to store data
        if 'employees_data' not in st.session_state:
            st.session_state.employees_data = []
        if 'projects_data' not in st.session_state:
            st.session_state.projects_data = []
        if 'tasks_data' not in st.session_state:
            st.session_state.tasks_data = []
        if 'documents_data' not in st.session_state:
            st.session_state.documents_data = []
        if 'notifications_data' not in st.session_state:
            st.session_state.notifications_data = []
        if 'issues_data' not in st.session_state:
            st.session_state.issues_data = []
            
        self.recommendation_model = None
        self.prediction_model = None
        self.skills_model = None
        
        # Workflow session states
        if 'current_project_workflow' not in st.session_state:
            st.session_state.current_project_workflow = None
        if 'workflow_step' not in st.session_state:
            st.session_state.workflow_step = "select_action"  # select_action -> edit_project -> recommendations -> assignment -> complete
        if 'project_form_data' not in st.session_state:
            st.session_state.project_form_data = {}
            
        # Task workflow session states
        if 'current_task_workflow' not in st.session_state:
            st.session_state.current_task_workflow = None
        if 'task_workflow_step' not in st.session_state:
            st.session_state.task_workflow_step = "select_action"  # select_action -> task_details -> predictions -> assignment -> complete
        if 'task_form_data' not in st.session_state:
            st.session_state.task_form_data = {}
        
        # Skill categories
        self.SKILL_CATEGORIES = {
            'Programming Language': ['Python', 'JavaScript', 'Java', 'C++', 'C#', 'Ruby', 'Go', 'Rust', 'PHP', 'Swift'],
            'Web Development': ['HTML', 'CSS', 'React', 'Vue.js', 'Angular', 'Node.js', 'Django', 'Flask', 'Spring', 'Express.js'],
            'Database & Cloud': ['SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'AWS', 'Docker', 'Kubernetes', 'Azure', 'GCP'],
            'Data Science': ['Machine Learning', 'TensorFlow', 'PyTorch', 'Data Analysis', 'Pandas', 'NumPy', 'R', 'Tableau'],
            'DevOps & Tools': ['Git', 'Jenkins', 'CI/CD', 'Linux', 'Bash', 'REST API', 'GraphQL', 'Microservices'],
            'Soft Skills': ['Project Management', 'Agile', 'Scrum', 'Communication', 'Leadership', 'Problem Solving']
        }
    
    # --- Core Functions ---
    
    def get_existing_ids(self):
        """Get all existing IDs to prevent duplicates"""
        employee_ids = [emp['employee_id'] for emp in st.session_state.employees_data]
        project_ids = [proj['project_id'] for proj in st.session_state.projects_data]
        task_ids = [task['task_id'] for task in st.session_state.tasks_data]
        return employee_ids, project_ids, task_ids
    
    def calculate_project_progress(self, project_id):
        """Calculate project progress based on tasks"""
        project_tasks = [task for task in st.session_state.tasks_data if task['project_id'] == project_id]
        if not project_tasks:
            return 0
        
        completed_tasks = len([task for task in project_tasks if task['status'] == 'Completed'])
        return int((completed_tasks / len(project_tasks)) * 100)
    
    def get_skill_match_recommendations(self, required_skills, min_match_threshold=0.6):
        """Get employee recommendations based on skill matching"""
        recommendations = []
        required_skills_list = [skill.strip() for skill in required_skills.split(';')] if required_skills else []
        
        for employee in st.session_state.employees_data:
            employee_skills = {}
            if 'skills' in employee and employee['skills']:
                for skill_item in employee['skills'].split('; '):
                    if ':' in skill_item:
                        skill, level = skill_item.split(':')
                        employee_skills[skill.strip()] = int(level)
            
            # Calculate skill match
            matched_skills = [skill for skill in required_skills_list if skill in employee_skills]
            match_ratio = len(matched_skills) / len(required_skills_list) if required_skills_list else 0
            
            if match_ratio >= min_match_threshold:
                avg_proficiency = sum([employee_skills[skill] for skill in matched_skills]) / len(matched_skills) if matched_skills else 0
                recommendations.append({
                    'employee': employee,
                    'match_ratio': match_ratio,
                    'matched_skills': matched_skills,
                    'avg_proficiency': avg_proficiency
                })
        
        # Sort by match ratio and proficiency
        recommendations.sort(key=lambda x: (x['match_ratio'], x['avg_proficiency']), reverse=True)
        return recommendations
    
    def add_notification(self, message, notification_type="info", project_id=None, task_id=None):
        """Add notification to the system"""
        notification_id = str(uuid.uuid4())[:8]
        st.session_state.notifications_data.append({
            'notification_id': notification_id,
            'message': message,
            'type': notification_type,
            'project_id': project_id,
            'task_id': task_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'read': False
        })

    def update_team_assignment(self, emp_id, action):
            """
            AI button callback: update data and force synchronization of Multiselect component state
            """
            # 1. Ensure data list exists
            if 'assigned_team' not in st.session_state.project_form_data:
                st.session_state.project_form_data['assigned_team'] = []
                
            current_team = st.session_state.project_form_data['assigned_team']
            
            # 2. Execute data update
            if action == "add":
                if emp_id not in current_team:
                    current_team.append(emp_id)
            elif action == "remove":
                if emp_id in current_team:
                    current_team.remove(emp_id)
            
            # 3. Save back to data source
            st.session_state.project_form_data['assigned_team'] = current_team
            
            # 4. [Critical Fix] Force update Multiselect component's Session State
            # Must build string list that exactly matches Multiselect option format ("ID - Name")
            emp_name_map = self.get_employee_name_map()
            formatted_selection = []
            for eid in current_team:
                name = emp_name_map.get(eid, 'Unknown')
                # Note: Format must exactly match multiselect options format
                formatted_selection.append(f"{eid} - {name}")
                
            # Update component's bound key so multiselect shows new data after page refresh
            st.session_state['team_selection'] = formatted_selection
    
    def _update_from_multiselect(self):
        """
        Multiselect component callback: when user manually selects in dropdown, sync back to data source
        """
        # Get selected full string list from component state
        selected_options = st.session_state.get('team_selection', [])
        
        # Extract IDs
        assigned_ids = [opt.split(" - ")[0] for opt in selected_options]
        
        # Update data source
        st.session_state.project_form_data['assigned_team'] = assigned_ids
    
    def get_employee_name_map(self):
        """Helper to get a dict of {id: name} for quick lookup"""
        name_map = {}
        for emp in st.session_state.employees_data:
            name_map[emp['employee_id']] = emp.get('name', 'Unknown')
        return name_map
    
    def get_project_team_members(self, project_id):
        """Get list of employee IDs assigned to a project"""
        project = next((p for p in st.session_state.projects_data if p['project_id'] == project_id), None)
        if project and 'assigned_team' in project:
            return project['assigned_team']
        return []
    
    # --- External Interface Methods ---

    def set_skills_model(self, model_function):
        """Set the external skills model interface"""
        self.skills_model = model_function
    
    def get_skills_from_external_model(self, project_data):
        """Get skills list from external skills model"""
        if self.skills_model is not None:
            try:
                skills_list = self.skills_model(project_data)
                st.session_state.current_recommendation_skills = skills_list
                return skills_list
            except Exception as e:
                st.warning(f"External skills model failed: {e}")
                return []
        else:
            # Fallback: extract from required_skillsets
            required_skills = project_data.get('required_skillsets', '')
            skills_list = [skill.strip() for skill in required_skills.split(';') if skill.strip()] if required_skills else []
            st.session_state.current_recommendation_skills = skills_list
            return skills_list
    
    def set_recommendation_model(self, model_function):
        """Set the external recommendation model interface"""
        self.recommendation_model = model_function

    def set_task_recommendation_model(self, model_function):
        """Set the external recommendation model interface"""
        self.task_recommendation_model = model_function
    
    def set_prediction_model(self, model_function):
        """Set the external prediction model interface"""
        self.prediction_model = model_function

    def get_task_recommendations(self,complexity):
        if self.task_recommendation_model is not None:
            return self.task_recommendation_model(complexity)
        
    def get_recommendations(self, required_skills, min_match_threshold=0.6):
        """Get recommendations using external model or fallback to internal matching"""
        if self.recommendation_model is not None:
            try:
                # Get project complexity (if available)
                project_complexity = None
                if hasattr(st.session_state, 'project_form_data') and st.session_state.project_form_data:
                    project_complexity = st.session_state.project_form_data.get('Complexity_Score')
                
                # Pass all necessary parameters to external model, including optional project_complexity
                return self.recommendation_model(
                    required_skills=required_skills,
                    employees_data=st.session_state.employees_data,
                    project_complexity=project_complexity,
                    min_match_threshold=min_match_threshold
                )
            except Exception as e:
                st.warning(f"External recommendation model failed: {e}. Using internal skill matching.")
                return self.get_skill_match_recommendations(required_skills, min_match_threshold)
        else:
            # Use internal skill matching as fallback
            return self.get_skill_match_recommendations(required_skills, min_match_threshold)
        
    def get_predictions(self, *args, **kwargs):
        """Get predictions using external model"""
        if hasattr(self, 'prediction_model'):
            return self.prediction_model(*args, **kwargs)
        else:
            st.warning("No prediction model set.")
            return None

    # --- Project Workflow Methods ---
    
    def start_project_workflow(self, project_id=None):
        """Start the project workflow"""
        st.session_state.current_project_workflow = project_id
        st.session_state.workflow_step = "edit_project"
        
        # If editing existing project, load data into form
        if project_id:
            project = next((p for p in st.session_state.projects_data if p['project_id'] == project_id), None)
            if project:
                st.session_state.project_form_data = project.copy()
        else:
            # If it's a new project, ensure previous form data is cleared
            st.session_state.project_form_data = {}
        st.rerun()
    
    def save_project_form_data(self, form_data):
        """Save form data to session state"""
        st.session_state.project_form_data = form_data
    
    def complete_project_workflow(self):
        """Complete the workflow and reset"""
        st.session_state.current_project_workflow = None
        st.session_state.workflow_step = "select_action"
        st.session_state.project_form_data = {}

    def _cancel_project_workflow(self):
        """Helper to cancel workflow and reset state"""
        # Clear current workflow state
        st.session_state.workflow_step = "select_action"
        st.session_state.current_project_workflow = None
        st.session_state.project_form_data = {}
        st.rerun()
    
    def render_project_workflow(self):
        """Render the main project workflow page"""
        st.header("📋 Project Management")
        
        # Workflow steps
        steps = ["1. Select Action", "2. Project Input", "3. Recommendations & Assignment", "4. Confirmation", "5. Complete"]
        current_step_index = {
            "select_action": 0,
            "edit_project": 1, 
            "recommendations": 2,
            "assignment": 3,
            "complete": 4
        }.get(st.session_state.workflow_step, 0)
        
        # Progress bar
        progress = current_step_index / (len(steps) - 1)
        st.progress(progress)
        
        # Step indicators
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if i == current_step_index:
                    st.markdown(f"**{step}** 🟢")
                elif i < current_step_index:
                    st.markdown(f"~~{step}~~ ✅")
                else:
                    st.markdown(f"{step} ⚪")
        
        st.divider()
        
        # Render current step
        if st.session_state.workflow_step == "select_action":
            self._render_project_select_action_step()
        elif st.session_state.workflow_step == "edit_project":
            self._render_edit_project_step()
        elif st.session_state.workflow_step == "recommendations":
            self._render_recommendations_step()
        elif st.session_state.workflow_step == "assignment":
            self._render_assignment_step()
        elif st.session_state.workflow_step == "complete":
            self._render_complete_step()
    
    def _render_project_select_action_step(self):
        """Render the action selection step with Import/Export and Quick Update"""
        st.subheader("1. Select Action")
        
        # Single column layout: Add → Edit/Update → Import/Export
        st.markdown("### 🆕 Add New Project")
        if st.button("Start New Project", use_container_width=True):
            self.start_project_workflow()
        
        st.divider()
        
        st.markdown("### ⚡ Manage Existing Project")
        
        if st.session_state.projects_data:
            # Add search and filter functionality
            col_search1, col_search2 = st.columns(2)
            with col_search1:
                search_term = st.text_input("🔍 Search projects", placeholder="Search by ID or name...", key="project_search")
            with col_search2:
                status_filter = st.selectbox("Filter by status", ["All Statuses"] + list(set(p['status'] for p in st.session_state.projects_data)), key="project_status_filter")
            
            # Filter projects
            filtered_projects = st.session_state.projects_data
            if search_term:
                search_lower = search_term.lower()
                filtered_projects = [p for p in filtered_projects if search_lower in p['project_id'].lower() or search_lower in p['project_name'].lower()]
            
            if status_filter != "All Statuses":
                filtered_projects = [p for p in filtered_projects if p['status'] == status_filter]
            
            if filtered_projects:
                project_options = [f"{p['project_id']} - {p['project_name']} ({p['status']})" for p in filtered_projects]
                selected_project_str = st.selectbox("Select Project to Manage", project_options)
                
                # --- NEW: Quick Action Area (Vertical Layout) ---
                if selected_project_str:
                    project_id = selected_project_str.split(" - ")[0]
                    # Find the project object directly
                    project_idx = next((i for i, p in enumerate(st.session_state.projects_data) if p['project_id'] == project_id), -1)
                    
                    if project_idx != -1:
                        project = st.session_state.projects_data[project_idx]
                        
                        with st.container(border=True):
                            st.markdown(f"**Selected:** {project['project_name']}")
                            
                            # --- 1. Full Edit Section ---
                            st.markdown("#### 📝 Detail Edit")
                            st.caption("Modify project details, team, and skills.")
                            if st.button("Edit Full Details", use_container_width=True, key=f"btn_edit_{project_id}"):
                                self.start_project_workflow(project_id)
                            
                            
                            # --- 2. Quick Status Update Section ---
                            st.markdown("#### 🚀 Quick Status Update")
                            status_options = ['Not Started', 'In Progress', 'Completed', 'On Hold']
                            current_status = project.get('status', 'Not Started')
                            if current_status not in status_options: status_options.append(current_status)
                            
                            new_status = st.selectbox("Update Status:", status_options, 
                                                    index=status_options.index(current_status),
                                                    key=f"quick_status_{project_id}")
                            
                            if st.button("Confirm Status Update", use_container_width=True, key=f"btn_upd_{project_id}"):
                                if new_status != current_status:
                                    # Direct Update
                                    st.session_state.projects_data[project_idx]['status'] = new_status
                                    self.add_notification(f"Project {project_id} status updated to {new_status}", "info", project_id)
                                    st.success(f"Updated to {new_status}!")
                                    st.rerun()
                                else:
                                    st.info("Status is already set to this value.")

                st.info(f"Found {len(filtered_projects)} project(s)")
            else:
                st.warning("No projects found matching your criteria")
        else:
            st.info("No projects available to edit")
        
        st.divider()
        
        # --- Import / Export Section ---
        st.subheader("💾 Project Data Import / Export")
        
        col_import, col_export = st.columns(2)
        
        # Export Section
        with col_export:
            st.write("**Export Project Data**")
            if st.session_state.projects_data:
                df_export = pd.DataFrame(st.session_state.projects_data)
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Projects as CSV",
                    data=csv_data,
                    file_name="projects_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No project data to export.")
        
        # Import Section
        with col_import:
            st.write("**Import Project Data (CSV)**")
            uploaded_file = st.file_uploader("Upload Project CSV", type=["csv"], key="import_project_csv")
            
            if uploaded_file is not None:
                try:
                    df_import = pd.read_csv(uploaded_file, keep_default_na=False)
                    st.dataframe(df_import.head(3), height=100, use_container_width=True)
                    
                    if st.button("Confirm Project Import", use_container_width=True):
                        _, project_ids, _ = self.get_existing_ids()
                        new_projects_added = 0
                        
                        for record in df_import.to_dict('records'):
                            if 'project_id' not in record or not record['project_id']: continue
                            if record['project_id'] in project_ids: continue
                                
                            record.setdefault('status', 'Not Started')
                            record.setdefault('manager_id', 'To be assigned')
                            record.setdefault('Priority', 'Medium')
                            record.setdefault('Complexity_Score', 5)
                            
                            if 'assigned_team' in record and isinstance(record['assigned_team'], str):
                                try:
                                    if record['assigned_team'].startswith('[') and record['assigned_team'].endswith(']'):
                                        import ast
                                        record['assigned_team'] = ast.literal_eval(record['assigned_team'])
                                    else:
                                        record['assigned_team'] = []
                                except:
                                    record['assigned_team'] = []
                            elif 'assigned_team' not in record or not isinstance(record['assigned_team'], list):
                                record['assigned_team'] = []
    
                            st.session_state.projects_data.append(record)
                            project_ids.append(record['project_id'])
                            new_projects_added += 1
                        
                        if new_projects_added > 0:
                            st.success(f"Successfully imported {new_projects_added} new projects!")
                            st.rerun()
                        else:
                            st.warning("No new projects imported. Check for duplicate IDs.")
                            
                except Exception as e:
                    st.error(f"Failed to import file: {e}")

    def _render_edit_project_step(self):
        """Render the project editing step (Fixed: Buttons on same line, No highlight)"""
        st.subheader("2. Edit Project Details")
        
        is_editing = st.session_state.current_project_workflow is not None
        
        with st.form("project_workflow_form"):

            col1, col2 = st.columns(2)
            
            employee_ids, project_ids, task_ids = self.get_existing_ids()
            
            with col1:
                if is_editing:
                    project_id = st.text_input("Project ID*", 
                                             value=st.session_state.project_form_data.get('project_id', ''),
                                             disabled=True)
                    st.info(f"Editing: {project_id}")
                else:
                    project_id = st.text_input("Project ID*", 
                                             value=st.session_state.project_form_data.get('project_id', ''),
                                             placeholder="PROJ-001")
                    if project_id and project_id in project_ids:
                        st.error("Project ID already exists!")
                
                project_name = st.text_input("Project Name*", 
                                           value=st.session_state.project_form_data.get('project_name', ''),
                                           placeholder="Quantum Platform Pro")
                start_date = st.date_input("Start Date*", 
                                         value=datetime.strptime(st.session_state.project_form_data.get('start_date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'))
                deadline = st.date_input("Deadline*", 
                                       value=datetime.strptime(st.session_state.project_form_data.get('deadline', (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')), '%Y-%m-%d'))
            
            with col2:
                complexity_score = st.slider("Complexity Score (1-10)*", 
                                           min_value=1, max_value=10, 
                                           value=st.session_state.project_form_data.get('Complexity_Score', 5))
                priority = st.selectbox("Priority*", 
                                      ['Low', 'Medium', 'High'],
                                      index=['Low', 'Medium', 'High'].index(st.session_state.project_form_data.get('Priority', 'Medium')))
                priority_score = {'Low': 1, 'Medium': 2, 'High': 3}[priority]
            
            description = st.text_area("Project Description*", 
                                     value=st.session_state.project_form_data.get('description', ''),
                                     placeholder="Develop a comprehensive digital platform...",
                                     height=100)
            
            st.subheader("Required Skillsets")
            current_skills = st.session_state.project_form_data.get('required_skillsets', '')
            current_skills_list = [s.strip() for s in current_skills.split(';')] if current_skills else []
            
            required_skills = []
            for category, skills in self.SKILL_CATEGORIES.items():
                with st.expander(f"{category}"):
                    default_skills = [s for s in skills if s in current_skills_list]
                    category_skills = st.multiselect(f"Select {category} skills", skills, default=default_skills, key=f"proj_skill_{category}")
                    required_skills.extend(category_skills)
            
            required_skills_string = "; ".join(required_skills)
            
            st.markdown("---")
            
            # --- Bottom button area (key modification) ---
            # 1. Ensure creating two columns inside Form
            c_cancel, c_continue = st.columns([1, 1]) 
            
            with c_cancel:
                # Cancel button: type="secondary" (default gray), full width
                cancel_btn = st.form_submit_button("❌ Cancel Workflow", type="secondary", use_container_width=True)
            
            with c_continue:
                # Continue button: type="secondary" (default gray), full width, removed highlight
                submit_btn = st.form_submit_button("➡️ Continue to Recommendations", type="secondary", use_container_width=True)
            
            # --- Logic judgment ---
            if cancel_btn:
                # Must check Cancel first so user can exit even without filling required fields
                self._cancel_project_workflow()
                
            elif submit_btn:
                # Only check required fields when clicking Continue
                if project_id and project_name and description:
                    if not is_editing and project_id in project_ids:
                        st.error("Project ID already exists!")
                    else:
                        form_data = {
                            'project_id': project_id,
                            'project_name': project_name,
                            'description': description,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'deadline': deadline.strftime('%Y-%m-%d'),
                            'status': st.session_state.project_form_data.get('status', 'Not Started'),
                            'manager_id': st.session_state.project_form_data.get('manager_id', 'To be assigned'),
                            'assigned_team': st.session_state.project_form_data.get('assigned_team', []),
                            'Complexity_Score': complexity_score,
                            'Priority': priority,
                            'Priority_Score': priority_score,
                            'required_skillsets': required_skills_string,
                            'created_date': st.session_state.project_form_data.get('created_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        self.save_project_form_data(form_data)
                        st.session_state.workflow_step = "recommendations"
                        st.rerun()
                else:
                    st.error("Please fill in all required fields (marked with *)")

    def _render_recommendations_step(self):
            """Render the recommendations step (Fixed version: bidirectional binding)"""
            st.subheader("3. Recommendations & Assignment")
            
            project_data = st.session_state.project_form_data
        
            # --- Prepare data mapping ---
            emp_name_map = self.get_employee_name_map()
            
            # --- Manual assignment section ---
            st.markdown("---")
            st.subheader("👥 Team Assignment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Assign Project Manager")
                if st.session_state.employees_data:
                    manager_options = [f"{e['employee_id']} - {e['name']} ({e['job_title']})" for e in st.session_state.employees_data]
                    current_manager = project_data.get('manager_id', 'To be assigned')
                    
                    manager_index = 0
                    if current_manager and current_manager != "To be assigned":
                        current_manager_option = f"{current_manager} - {emp_name_map.get(current_manager, 'Unknown')}"
                        # Simple fault-tolerant matching
                        match = next((opt for opt in manager_options if opt.startswith(f"{current_manager} -")), None)
                        if match:
                            manager_index = manager_options.index(match) + 1
                        else:
                            manager_options = [current_manager_option] + manager_options
                            manager_index = 1
                    
                    selected_manager = st.selectbox("Select Project Manager", [""] + manager_options, index=manager_index, key="manager_selection")
                    if selected_manager:
                        project_data['manager_id'] = selected_manager.split(" - ")[0]
                    else:
                        project_data['manager_id'] = "To be assigned"
                else:
                    st.info("No employees available")
            
            with col2:
                st.markdown("### Assign Team Members")
                if st.session_state.employees_data:
                    # 1. Build options list
                    # Format here must be consistent with format built in update_team_assignment!
                    employee_options = [f"{e['employee_id']} - {e['name']}" for e in st.session_state.employees_data]
                    
                    # 2. Initialize component state (if not yet initialized)
                    # This ensures correct multiselect display on first page load
                    current_team_ids = project_data.get('assigned_team', [])
                    if 'team_selection' not in st.session_state:
                        initial_selection = []
                        for eid in current_team_ids:
                            name = emp_name_map.get(eid, 'Unknown')
                            val = f"{eid} - {name}"
                            if val in employee_options:
                                initial_selection.append(val)
                        st.session_state['team_selection'] = initial_selection
    
                    # 3. Render component
                    # Note: Using on_change callback, not default
                    # And completely removed assignment logic below component
                    st.multiselect(
                        "Select Team Members", 
                        options=employee_options, 
                        key="team_selection",            # Bind to session_state['team_selection']
                        on_change=self._update_from_multiselect  # Trigger callback when manually modified
                    )
                    
                    if current_team_ids:
                        st.info(f"Selected {len(current_team_ids)} team members")
                else:
                    st.info("No employees available")
            
            # --- AI recommendation section ---
            st.markdown("---")
            st.subheader("🤖 AI Recommendations")
            
            if st.session_state.employees_data:
                try:
                    recommendations = self.get_recommendations(project_data['required_skillsets'])
                except Exception as e:
                    st.error(f"Error getting recommendations: {e}")
                    recommendations = []
                
                if recommendations:
                    st.markdown("**Top AI Recommendations based on skill matching:**")
                    
                    # Get latest ID list for judging button state
                    current_team_ids = project_data.get('assigned_team', [])
                    
                    # Display employee recommendations first
                    for i, rec in enumerate(recommendations[:5]):
                        employee = rec['employee']
                        emp_id = employee['employee_id']
                        is_assigned = emp_id in current_team_ids
                        
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([3, 2, 1])
                            
                            with col1:
                                st.markdown(f"**{employee['name']}**")
                                st.markdown(f"*{employee['job_title']}* | {employee['department']}")
                            
                            with col2:
                                st.markdown(f"**Match: {rec.get('match_ratio', 0):.0%}**")
                                st.markdown(f"Skills: {', '.join(rec['matched_skills'][:3])}")
                            
                            with col3:
                                # Use on_click callback, this triggers update_team_assignment
                                if is_assigned:
                                    st.button("✅ Remove", 
                                            key=f"remove_ai_{emp_id}_{i}", 
                                            use_container_width=True,
                                            on_click=self.update_team_assignment,
                                            args=(emp_id, "remove"))
                                else:
                                    st.button("➕ Add", 
                                            key=f"add_ai_{emp_id}_{i}", 
                                            use_container_width=True,
                                            type="primary", 
                                            on_click=self.update_team_assignment,
                                            args=(emp_id, "add"))
                    
                    # --- NEW: Display skills after employee recommendations ---
                    # Get skills from external skills model
                    skills_list = self.get_skills_from_external_model(project_data['description'])
                    
                    if skills_list:
                        #st.markdown("---")
                        #st.markdown(f"**🔍 Recommendation Basis**")
                        
                        # tight badge layout
                        st.write("Skills used for matching:")
                        
                        # create a row of badges
                        badge_container = st.container()
                        with badge_container:
                            cols = st.columns([1] * min(len(skills_list), 8))  # 8 badges at most
                            for i, skill in enumerate(skills_list):
                                if i < len(cols):
                                    with cols[i]:
                                        st.badge(skill)
                
                else:
                    st.info("No recommendations found.")
                    if not project_data.get('required_skillsets'):
                        st.warning("No required skills specified for this project. To get AI recommendations, please go back and add the corresponding information.")
                
            # --- Bottom navigation buttons ---
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1]) # Change to three columns
            
            with col1:
                if st.button("← Back", use_container_width=True):
                    st.session_state.workflow_step = "edit_project"
                    st.rerun()
                    
            with col2:
                # Put Cancel in middle
                if st.button("❌ Cancel", key="cancel_step_3", use_container_width=True):
                    self._cancel_project_workflow()
                    
            with col3:
                if st.button("Continue →", use_container_width=True):
                    if not st.session_state.project_form_data.get('assigned_team'):
                        st.warning("No team members assigned. Are you sure you want to continue?")
                    st.session_state.workflow_step = "assignment"
                    st.rerun()

    def _render_assignment_step(self):
        """Render the final assignment step"""
        st.subheader("4. Finalize Project Assignment")
        
        project_data = st.session_state.project_form_data
        
        # Project summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Project Details")
            st.write(f"**ID:** {project_data.get('project_id')}")
            st.write(f"**Name:** {project_data.get('project_name')}")
            st.write(f"**Description:** {project_data.get('description')}")
            st.write(f"**Priority:** {project_data.get('Priority')}")
            st.write(f"**Complexity:** {project_data.get('Complexity_Score')}/10")
        
        with col2:
            st.markdown("### Team Assignment")
            emp_name_map = self.get_employee_name_map()
            st.write(f"**Project Manager:** {emp_name_map.get(project_data.get('manager_id'), project_data.get('manager_id', 'Not assigned'))}")
            
            st.markdown("**Assigned Team:**")
            if project_data.get('assigned_team'):
                for emp_id in project_data['assigned_team']:
                    st.write(f"- {emp_name_map.get(emp_id, emp_id)}")
                st.info(f"Total team members: {len(project_data['assigned_team'])}")
            else:
                st.write("No team assigned")
            
            st.markdown(f"**Required Skills:** {project_data.get('required_skillsets', 'None')}")
    
        
        st.divider()
        
        # Final actions
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back"):
                st.session_state.workflow_step = "recommendations"
                st.rerun()
        
        with col2:
            # Cancel button
            if st.button("❌ Cancel", key="cancel_step_4"):
                self._cancel_project_workflow()
        
        with col3:
            if st.button("🚀 Confirm", type="primary"):
                # Save to projects data
                project_id = self._save_project_data(project_data, is_draft=False)
                if project_id:
                    st.session_state.workflow_step = "complete"
                    st.rerun()
    
    def _save_project_data(self, project_data, is_draft=False):
        """Save project data to the main projects list"""
        employee_ids, project_ids, task_ids = self.get_existing_ids()
        
        if project_data['project_id'] in project_ids:
            # Update existing project
            index = next((i for i, p in enumerate(st.session_state.projects_data) if p['project_id'] == project_data['project_id']), -1)
            if index >= 0:
                st.session_state.projects_data[index] = project_data
                status = "updated"
        else:
            # Add new project
            if is_draft:
                project_data['status'] = 'Draft'
            st.session_state.projects_data.append(project_data)
            status = "created"
        
        project_id = project_data['project_id']
        self.add_notification(f"Project {project_data['project_name']} {status}", "info", project_id)
        return project_id
    
    def _render_complete_step(self):
        """Render the completion step"""
        st.subheader("🎉 Project Setup Complete!")
        
        project_data = st.session_state.project_form_data
        
        st.success(f"Project **{project_data.get('project_name')}** has been successfully created/updated!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Next Steps")
            st.markdown("""
            - 📋 **Manage Tasks**: Add tasks to your new project
            - 👥 **Team Coordination**: Coordinate with assigned team members  
            - 📊 **Track Progress**: Monitor project progress in Analytics
            - 📁 **Upload Documents**: Add project documentation
            """)
        
        with col2:
            st.markdown("### Quick Action")
            if st.button("🆕 Create/Edit Another Project"):
                self.complete_project_workflow()
                st.rerun()

    # --- Task Workflow Methods ---
    
    def start_task_workflow(self, task_id=None):
        """Start the task workflow"""
        st.session_state.current_task_workflow = task_id
        st.session_state.task_workflow_step = "task_details"
        
        # If editing existing task, load data into form
        if task_id:
            task = next((t for t in st.session_state.tasks_data if t['task_id'] == task_id), None)
            if task:
                st.session_state.task_form_data = task.copy()
        else:
            # If it's a new task, ensure previous form data is cleared
            st.session_state.task_form_data = {}

        st.rerun()
    
    def save_task_form_data(self, form_data):
        """Save task form data to session state"""
        st.session_state.task_form_data = form_data
    
    def complete_task_workflow(self):
        """Complete the task workflow and reset"""
        st.session_state.current_task_workflow = None
        st.session_state.task_workflow_step = "select_action"
        st.session_state.task_form_data = {}

    def _cancel_task_workflow(self):
        """Helper to cancel task workflow and reset state"""
        st.session_state.task_workflow_step = "select_action"
        st.session_state.current_task_workflow = None
        st.session_state.task_form_data = {}
        st.rerun()
    
    def render_task_workflow(self):
        """Render the main task workflow page"""
        st.header("✅ Task Management")
        
        # Workflow steps
        steps = ["1. Select Action", "2. Task Details", "3. Predictions & Assignment", "4. Confirmation", "5. Complete"]
        current_step_index = {
            "select_action": 0,
            "task_details": 1, 
            "predictions": 2,
            "assignment": 3,
            "complete": 4
        }.get(st.session_state.task_workflow_step, 0)
        
        # Progress bar
        progress = current_step_index / (len(steps) - 1)
        st.progress(progress)
        
        # Step indicators
        cols = st.columns(len(steps))
        for i, step in enumerate(steps):
            with cols[i]:
                if i == current_step_index:
                    st.markdown(f"**{step}** 🟢")
                elif i < current_step_index:
                    st.markdown(f"~~{step}~~ ✅")
                else:
                    st.markdown(f"{step} ⚪")
        
        st.divider()
        
        # Render current step
        if st.session_state.task_workflow_step == "select_action":
            self._render_task_select_action_step()
        elif st.session_state.task_workflow_step == "task_details":
            self._render_task_details_step()
        elif st.session_state.task_workflow_step == "predictions":
            self._render_task_predictions_step()
        elif st.session_state.task_workflow_step == "assignment":
            self._render_task_assignment_step()
        elif st.session_state.task_workflow_step == "complete":
            self._render_task_complete_step()
    
    def _render_task_select_action_step(self):
        """Render the task action selection step with Import/Export and Quick Update"""
        st.subheader("1. Select Action")
        
        # Single column layout: Add → Edit/Update → Import/Export
        st.markdown("### 🆕 Add New Task")
        if st.button("Start New Task", use_container_width=True):
            self.start_task_workflow()
        
        st.divider()
        
        st.markdown("### ⚡ Manage Existing Task")
        
        if st.session_state.tasks_data:
            # Add search and filter functionality
            col_search1, col_search2, col_search3 = st.columns(3)
            with col_search1:
                search_term = st.text_input("🔍 Search tasks", placeholder="Search by ID or name...", key="task_search_select")
            with col_search2:
                status_filter = st.selectbox("Filter by status", ["All Statuses"] + list(set(t['status'] for t in st.session_state.tasks_data)), key="task_status_filter_select")
            with col_search3:
                project_options = ["All Projects"] + list(set(t['project_id'] for t in st.session_state.tasks_data))
                project_filter = st.selectbox("Filter by project", project_options, key="task_project_filter_select")
            
            # Filter tasks
            filtered_tasks = st.session_state.tasks_data
            if search_term:
                search_lower = search_term.lower()
                filtered_tasks = [t for t in filtered_tasks if search_lower in t['task_id'].lower() or search_lower in t['task_name'].lower()]
            
            if status_filter != "All Statuses":
                filtered_tasks = [t for t in filtered_tasks if t['status'] == status_filter]
            
            if project_filter != "All Projects":
                filtered_tasks = [t for t in filtered_tasks if t['project_id'] == project_filter]
            
            if filtered_tasks:
                task_options = [f"{t['task_id']} - {t['task_name']} ({t['status']})" for t in filtered_tasks]
                selected_task_str = st.selectbox("Select Task to Manage", task_options)
                
                # --- NEW: Quick Action Area (Vertical Layout) ---
                if selected_task_str:
                    task_id = selected_task_str.split(" - ")[0]
                    task_idx = next((i for i, t in enumerate(st.session_state.tasks_data) if t['task_id'] == task_id), -1)
                    
                    if task_idx != -1:
                        task = st.session_state.tasks_data[task_idx]
                        
                        with st.container(border=True):
                            st.markdown(f"**Selected:** {task['task_name']}")
                            
                            # --- 1. Full Edit Section ---
                            st.markdown("#### 📝 Detail Edit")
                            st.caption("Edit description, assignment, and estimates.")
                            if st.button("Edit Full Details", use_container_width=True, key=f"btn_edit_task_{task_id}"):
                                self.start_task_workflow(task_id)
                            
                            # --- 2. Quick Status Update Section ---
                            st.markdown("#### 🚀 Quick Status Update")
                            status_options = ['Unassigned', 'Assigned', 'In Progress', 'Completed', 'Blocked']
                            current_status = task.get('status', 'Unassigned')
                            if current_status not in status_options: status_options.append(current_status)
                            
                            new_status = st.selectbox("Update Status:", status_options, 
                                                    index=status_options.index(current_status),
                                                    key=f"quick_status_task_{task_id}")
                            
                            if st.button("Confirm Status Update", use_container_width=True, key=f"btn_upd_task_{task_id}"):
                                if new_status != current_status:
                                    # Direct Update
                                    st.session_state.tasks_data[task_idx]['status'] = new_status
                                    self.add_notification(f"Task {task_id} status updated to {new_status}", "info", task.get('project_id'), task_id)
                                    st.success(f"Updated to {new_status}!")
                                    st.rerun()
                                else:
                                    st.info("Status is already set to this value.")

                st.info(f"Found {len(filtered_tasks)} task(s)")
            else:
                st.warning("No tasks found matching your criteria")
        else:
            st.info("No tasks available to edit")
        
        st.divider()
        
        # --- Import / Export Section ---
        st.subheader("💾 Task Data Import / Export")
        
        col_import, col_export = st.columns(2)
        
        # Export Section
        with col_export:
            st.write("**Export Task Data**")
            if st.session_state.tasks_data:
                df_export = pd.DataFrame(st.session_state.tasks_data)
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Tasks as CSV",
                    data=csv_data,
                    file_name="tasks_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No task data to export.")
        
        # Import Section
        with col_import:
            st.write("**Import Task Data (CSV)**")
            uploaded_file = st.file_uploader("Upload Task CSV", type=["csv"], key="import_task_csv_workflow")
            
            if uploaded_file is not None:
                try:
                    df_import = pd.read_csv(uploaded_file, keep_default_na=False)
                    st.dataframe(df_import.head(3), height=100, use_container_width=True)
                    
                    if st.button("Confirm Task Import", use_container_width=True):
                        _, _, task_ids = self.get_existing_ids()
                        new_tasks_added = 0
                        
                        for record in df_import.to_dict('records'):
                            if 'task_id' not in record or not record['task_id']: continue
                            if record['task_id'] in task_ids: continue
                                
                            record.setdefault('status', 'Unassigned')
                            record.setdefault('category', 'Other')
                            record.setdefault('complexity', 3)
                            record.setdefault('estimated_duration', 5)
                            record.setdefault('estimated_budget', 1000)
                            record.setdefault('start_date', datetime.now().strftime('%Y-%m-%d'))
                            
                            st.session_state.tasks_data.append(record)
                            task_ids.append(record['task_id'])
                            new_tasks_added += 1
                        
                        if new_tasks_added > 0:
                            st.success(f"Successfully imported {new_tasks_added} new tasks!")
                            st.rerun()
                        else:
                            st.warning("No new tasks imported. Check for duplicate IDs.")
                            
                except Exception as e:
                    st.error(f"Failed to import file: {e}")

    def _render_task_details_step(self):
        """Render the task details step"""
        st.subheader("2. Task Details")
        
        is_editing = st.session_state.current_task_workflow is not None
        
        with st.form("task_workflow_form"):
            col1, col2 = st.columns(2)
            
            employee_ids, project_ids, task_ids = self.get_existing_ids()
            
            with col1:
                if is_editing:
                    task_id = st.text_input("Task ID*", 
                                          value=st.session_state.task_form_data.get('task_id', ''),
                                          disabled=True)
                    st.info(f"Editing: {task_id}")
                else:
                    task_id = st.text_input("Task ID*", 
                                          value=st.session_state.task_form_data.get('task_id', ''),
                                          placeholder="TASK-1001")
                    if task_id and task_id in task_ids:
                        st.error("Task ID already exists!")
                
                task_name = st.text_input("Task Name*", 
                                        value=st.session_state.task_form_data.get('task_name', ''),
                                        placeholder="Implement Authentication System")
                
                # Project selection
                if st.session_state.projects_data:
                    project_options = [f"{p['project_id']} - {p['project_name']}" for p in st.session_state.projects_data]
                    current_project = st.session_state.task_form_data.get('project_id', '')
                    
                    project_index = 0
                    if current_project:
                        current_project_option = f"{current_project} - {next((p['project_name'] for p in st.session_state.projects_data if p['project_id'] == current_project), 'Unknown')}"
                        match = next((opt for opt in project_options if opt.startswith(f"{current_project} -")), None)
                        if match:
                            project_index = project_options.index(match)
                    
                    selected_project = st.selectbox("Project*", project_options, index=project_index)
                    project_id = selected_project.split(" - ")[0]
                else:
                    st.error("No projects available. Please create a project first.")
                    project_id = ""

                                # Fix category selection error
                category_options = ['Frontend', 'Backend', 'Database', 'DevOps', 'Security', 'Design', 'Testing', 'Other']
                current_category = st.session_state.task_form_data.get('category', 'Other')
                
                # Safely get index
                try:
                    category_index = category_options.index(current_category)
                except ValueError:
                    category_index = category_options.index('Other')  # Default value
                
                category = st.selectbox("Category*", category_options, index=category_index)
            
            with col2:
                complexity = st.slider("Complexity (1-5)*", min_value=1, max_value=5, 
                                     value=st.session_state.task_form_data.get('complexity', 3))
                
                
                # Add Start Date
                start_date = st.date_input("Start Date*", 
                                         value=datetime.strptime(st.session_state.task_form_data.get('start_date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'))
                
                deadline = st.date_input("Deadline*", 
                                       value=datetime.strptime(st.session_state.task_form_data.get('deadline', (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')), '%Y-%m-%d'))
            
            task_description = st.text_area("Task Description*", 
                                          value=st.session_state.task_form_data.get('task_description', ''),
                                          placeholder="Describe the task in detail...",
                                          height=100)
            
            st.subheader("Required Skillsets")
            current_skills = st.session_state.task_form_data.get('required_skillsets', '')
            current_skills_list = [s.strip() for s in current_skills.split(';')] if current_skills else []
            
            required_skills = []
            for skill_category, skills in self.SKILL_CATEGORIES.items():
                with st.expander(f"{skill_category}"):
                    default_skills = [s for s in skills if s in current_skills_list]
                    category_skills = st.multiselect(f"Select {skill_category} skills", skills, default=default_skills, key=f"task_skill_{skill_category}")
                    required_skills.extend(category_skills)
            
            required_skills_string = "; ".join(required_skills)
            
            st.markdown("---")
            
            # Fix: Add form submit button
            submit_col1, submit_col2 = st.columns([1, 1])
            
            with submit_col1:
                cancel_btn = st.form_submit_button("❌ Cancel Workflow", type="secondary", use_container_width=True)
            
            with submit_col2:
                submit_btn = st.form_submit_button("➡️ Continue to Predictions", type="secondary", use_container_width=True)
            
            if cancel_btn:
                self._cancel_task_workflow()
                
            elif submit_btn:
                if task_id and task_name and task_description and project_id:
                    if not is_editing and task_id in task_ids:
                        st.error("Task ID already exists!")
                    else:
                        form_data = {
                            'task_id': task_id,
                            'project_id': project_id,
                            'task_name': task_name,
                            'task_description': task_description,
                            'complexity': complexity,
                            'category': category,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'deadline': deadline.strftime('%Y-%m-%d'),
                            'status': st.session_state.task_form_data.get('status', 'Unassigned'),
                            'assigned_to_id': st.session_state.task_form_data.get('assigned_to_id', ''),
                            'assigned_to_name': st.session_state.task_form_data.get('assigned_to_name', ''),
                            'required_skillsets': required_skills_string,
                            'estimated_duration': st.session_state.task_form_data.get('estimated_duration', 5),
                            'estimated_budget': st.session_state.task_form_data.get('estimated_budget', 1000),
                            'created_date': st.session_state.task_form_data.get('created_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        self.save_task_form_data(form_data)
                        st.session_state.task_workflow_step = "predictions"
                        st.rerun()
                else:
                    st.error("Please fill in all required fields (marked with *)")

    def _render_task_predictions_step(self):
        """Render the task predictions step"""
        st.subheader("3. Predictions & Assignment")
        
        task_data = st.session_state.task_form_data
        
        # Get AI predictions for duration and budget
        category = task_data.get('category', 'Other')
        complexity = task_data.get('complexity', 3)
        
        # Use internal recommendation system
        rec_duration, rec_budget = self.get_task_recommendations(complexity)
        
        st.markdown("### 🤖 AI Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Duration", f"{rec_duration} days")
            estimated_duration = st.number_input("Estimated Duration (Days)*", 
                                              min_value=1, max_value=365, 
                                              value=rec_duration)
        
        with col2:
            st.metric("Predicted Budget", f"${rec_budget:,.0f}")
            estimated_budget = st.number_input("Estimated Budget (USD)*", 
                                            min_value=0, value=rec_budget, step=100)
        
        # Update task data with predictions
        task_data['estimated_duration'] = estimated_duration
        task_data['estimated_budget'] = estimated_budget
        
        st.info(f"**Recommendation based on '{category}' (Complexity: {complexity}):** \n"
                f"* Estimated Duration: **{rec_duration} days** \n"
                f"* Estimated Budget: **${rec_budget:,.0f}**")
        
        # Employee assignment
        st.markdown("---")
        st.subheader("👤 Assign Employee")
        
        project_id = task_data.get('project_id')
        if project_id:
            project_team_ids = self.get_project_team_members(project_id)
            emp_name_map = self.get_employee_name_map()
            
            if project_team_ids:
                employee_options = [f"{eid} - {emp_name_map.get(eid, 'Unknown')}" for eid in project_team_ids]
                
                current_assignee = task_data.get('assigned_to_id', '')
                assignee_index = 0
                if current_assignee:
                    match = next((opt for opt in employee_options if opt.startswith(f"{current_assignee} -")), None)
                    if match:
                        assignee_index = employee_options.index(match) + 1
                
                selected_employee = st.selectbox("Assign to Employee", [""] + employee_options, index=assignee_index)
                
                if selected_employee:
                    task_data['assigned_to_id'] = selected_employee.split(" - ")[0]
                    task_data['assigned_to_name'] = selected_employee.split(" - ")[1]
                    # Auto-update status if assigned
                    if task_data.get('status') == 'Unassigned':
                        task_data['status'] = 'Assigned'
                else:
                    task_data['assigned_to_id'] = ''
                    task_data['assigned_to_name'] = ''
            else:
                st.warning(f"No team members assigned to project {project_id}. Please assign team members in Project Management first.")
        else:
            st.error("No project selected. Please go back and select a project.")
        
        # Navigation buttons
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.task_workflow_step = "task_details"
                st.rerun()
                
        with col2:
            if st.button("❌ Cancel", key="cancel_task_step_3", use_container_width=True):
                self._cancel_task_workflow()
                
        with col3:
            if st.button("Continue →", use_container_width=True):
                st.session_state.task_workflow_step = "assignment"
                st.rerun()

    def _render_task_assignment_step(self):
        """Render the final task assignment step"""
        st.subheader("4. Finalize Task Assignment")
        
        task_data = st.session_state.task_form_data
        
        # Task summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Task Details")
            st.write(f"**ID:** {task_data.get('task_id')}")
            st.write(f"**Name:** {task_data.get('task_name')}")
            st.write(f"**Project:** {task_data.get('project_id')}")
            st.write(f"**Description:** {task_data.get('task_description')}")
            st.write(f"**Category:** {task_data.get('category')}")
            st.write(f"**Complexity:** {task_data.get('complexity')}/5")
        
        with col2:
            st.markdown("### Assignment & Predictions")
            emp_name_map = self.get_employee_name_map()
            assigned_to = task_data.get('assigned_to_id', 'Not assigned')
            assigned_name = emp_name_map.get(assigned_to, 'Not assigned')
            
            st.write(f"**Assigned To:** {assigned_name}")
            st.write(f"**Status:** {task_data.get('status', 'Unassigned')}")
            st.write(f"**Estimated Duration:** {task_data.get('estimated_duration')} days")
            st.write(f"**Estimated Budget:** ${task_data.get('estimated_budget'):,.0f}")
            st.write(f"**Deadline:** {task_data.get('deadline')}")
            
            if task_data.get('required_skillsets'):
                st.write(f"**Required Skills:** {task_data.get('required_skillsets')}")
        
        st.divider()
        
        # Final actions
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("← Back"):
                st.session_state.task_workflow_step = "predictions"
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", key="cancel_task_step_4"):
                self._cancel_task_workflow()
        
        with col3:
            if st.button("🚀 Confirm", type="primary"):
                # Save to tasks data
                task_id = self._save_task_data(task_data)
                if task_id:
                    st.session_state.task_workflow_step = "complete"
                    st.rerun()

    def _save_task_data(self, task_data):
        """Save task data to the main tasks list"""
        employee_ids, project_ids, task_ids = self.get_existing_ids()
        
        if task_data['task_id'] in task_ids:
            # Update existing task
            index = next((i for i, t in enumerate(st.session_state.tasks_data) if t['task_id'] == task_data['task_id']), -1)
            if index >= 0:
                st.session_state.tasks_data[index] = task_data
                status = "updated"
        else:
            # Add new task
            st.session_state.tasks_data.append(task_data)
            status = "created"
        
        task_id = task_data['task_id']
        self.add_notification(f"Task {task_data['task_name']} {status}", "info", task_data.get('project_id'), task_id)
        return task_id

    def _render_task_complete_step(self):
        """Render the task completion step"""
        st.subheader("🎉 Task Setup Complete!")
        
        task_data = st.session_state.task_form_data
        
        st.success(f"Task **{task_data.get('task_name')}** has been successfully created/updated!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Next Steps")
            st.markdown("""
            - 📋 **Manage Tasks**: View and manage all tasks
            - 👥 **Team Coordination**: Coordinate with assigned team member  
            - 📊 **Track Progress**: Monitor task progress in Analytics
            - 📁 **Upload Documents**: Add task-related documentation
            """)
        
        with col2:
            st.markdown("### Quick Action")
            if st.button("🆕 Create/Edit Another Task"):
                self.complete_task_workflow()
                st.rerun()

    # --- UI Components ---
    
    def render_dashboard(self):
        """Render Dashboard Tab"""
        st.header("📊 Project Dashboard")
        
        # Key metrics
        total_projects = len(st.session_state.projects_data)
        total_employees = len(st.session_state.employees_data)
        total_tasks = len(st.session_state.tasks_data)
        #completed_tasks = len([t for t in st.session_state.tasks_data if t['status'] == 'Completed'])
        open_issues = len([i for i in st.session_state.issues_data if i['status'] in ['Open', 'In Progress', 'Blocked']])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Projects", total_projects)
        with col2:
            st.metric("Total Employees", total_employees)
        with col3:
            st.metric("Total Tasks", total_tasks)
        #with col4:
        #    st.metric("Tasks Completed", f"{completed_tasks}/{total_tasks}" if total_tasks > 0 else "0/0")
        with col4:
            st.metric("Open Issues", open_issues)
        
        # Project progress overview
        st.subheader("Project Progress Overview")
        if st.session_state.projects_data:
            progress_data = []
            for project in st.session_state.projects_data:
                progress = self.calculate_project_progress(project['project_id'])
                progress_data.append({
                    'Project ID': project['project_id'],
                    'Project Name': project['project_name'],
                    'Status': project['status'],
                    'Progress': progress,
                    'Priority': project['Priority'],
                })
            
            progress_df = pd.DataFrame(progress_data)
            st.dataframe(progress_df, use_container_width=True)
            
            # Progress visualization
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Project Status Distribution")
                status_counts = pd.Series([p['status'] for p in st.session_state.projects_data]).value_counts()
                st.bar_chart(status_counts)
            
            with col2:
                st.subheader("Priority Distribution")
                priority_counts = pd.Series([p['Priority'] for p in st.session_state.projects_data]).value_counts()
                st.bar_chart(priority_counts)
        else:
            st.info("No projects available. Add projects in the Project Management tab.")
        
        # Recent notifications
        st.subheader("🔔 Recent Notifications")
        recent_notifications = st.session_state.notifications_data[-5:] if st.session_state.notifications_data else []
        for notification in reversed(recent_notifications):
            emoji = "ℹ️" if notification['type'] == 'info' else "⚠️" if notification['type'] == 'warning' else "🚨"
            st.write(f"{emoji} {notification['timestamp']}: {notification['message']}")

    def render_employee_management(self):
        """Render Employee Management Tab"""
        st.header("👥 Employee Management")
        
        with st.form("employee_form"):
            col1, col2 = st.columns(2)
            
            employee_ids, project_ids, task_ids = self.get_existing_ids()
            
            with col1:
                employee_id = st.text_input("Employee ID*", placeholder="EMP-001")
                
                if employee_id and employee_id in employee_ids:
                    st.error("Employee ID already exists! Please use a unique ID.")
                
                name = st.text_input("Name*", placeholder="John Smith")
                job_title = st.selectbox("Job Title*", [
                    'Software Engineer', 'Senior Software Engineer', 'Frontend Developer', 'Backend Developer',
                    'Full Stack Developer', 'Data Scientist', 'DevOps Engineer', 'Product Manager',
                    'UI/UX Designer', 'Data Analyst', 'System Architect', 'Project Manager'
                ])
                department = st.selectbox("Department*", ['Engineering', 'Data Science', 'Product', 'Design', 'DevOps', 'Other'])
            
            with col2:
                experience_years = st.number_input("Experience (Years)*", min_value=0, max_value=50, value=3)
                performance_rating = st.slider("Performance Rating*", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
                email = st.text_input("Email*", placeholder="john.smith@company.com")
            
            st.subheader("Skills & Proficiencies")
            selected_skills = {}
            for category, skills in self.SKILL_CATEGORIES.items():
                if category and skills:
                    with st.expander(f"{category}"):
                        # Use 5-column layout inside expander
                        num_columns = 5
                        columns = st.columns(num_columns)
                        
                        # Evenly distribute skills across 5 columns
                        for idx, skill in enumerate(skills):
                            if skill:
                                col_idx = idx % num_columns
                                with columns[col_idx]:
                                    # Checkbox on top
                                    if st.checkbox(f"Add {skill}", key=f"emp_skill_{skill}"):
                                        # Slider below - always shown but only saved when checked
                                        proficiency = st.slider(
                                            f"{skill} Proficiency", 
                                            1, 5, 
                                            1,
                                            key=f"emp_prof_{skill}"
                                        )
                                        selected_skills[skill] = proficiency
                                    else:
                                        # Also show slider when not checked, but don't save
                                        st.slider(
                                            f"{skill} Proficiency", 
                                            1, 5, 
                                            1,
                                            key=f"emp_prof_disabled_{skill}"
                                        )
            
            skills_string = "; ".join([f"{skill}:{level}" for skill, level in selected_skills.items()])
            
            submitted = st.form_submit_button("Add Employee")
            
            if submitted:
                if employee_id and name and email:
                    if employee_id in employee_ids:
                        st.error("Employee ID already exists! Please use a unique ID.")
                    else:
                        employee_data = {
                            'employee_id': employee_id,
                            'name': name,
                            'job_title': job_title,
                            'experience_years': experience_years,
                            'performance_rating': performance_rating,
                            'email': email,
                            'department': department,
                            'skills': skills_string,
                            'total_skills': len(selected_skills),
                            'avg_proficiency': round(sum(selected_skills.values()) / len(selected_skills), 1) if selected_skills else 0
                        }
                        st.session_state.employees_data.append(employee_data)
                        self.add_notification(f"New employee added: {name} ({employee_id})", "info")
                        st.success(f"Employee {name} added successfully!")
                else:
                    st.error("Please fill in all required fields (marked with *)")
        
        st.divider()
        
        # Employee list with editing capability
        if st.session_state.employees_data:
            # Add search and filter functionality
            st.subheader("Employee Search & Management")
            
            # Search and filter controls
            col_search1, col_search2, col_search3 = st.columns([2, 2, 1])
            with col_search1:
                search_term = st.text_input("🔍 Search by name, ID, or job title", placeholder="Enter search term...")
            with col_search2:
                department_filter = st.selectbox("Filter by department", ["All Departments"] + list(set(emp['department'] for emp in st.session_state.employees_data)))
            with col_search3:
                show_all = st.checkbox("Show All", value=False, key="show_all_employees")
            
            # Filter employees
            filtered_employees = st.session_state.employees_data
            
            if search_term:
                search_lower = search_term.lower()
                filtered_employees = [
                    emp for emp in filtered_employees 
                    if (search_lower in emp['name'].lower() or 
                        search_lower in emp['employee_id'].lower() or 
                        search_lower in emp['job_title'].lower())
                ]
            
            if department_filter != "All Departments":
                filtered_employees = [emp for emp in filtered_employees if emp['department'] == department_filter]
            
            # Display search result statistics
            if search_term or department_filter != "All Departments":
                st.info(f"Found {len(filtered_employees)} employee(s) matching your criteria")
            
            if not show_all and not search_term and department_filter == "All Departments":
                st.info("Use the search box or filters to view employees, or check 'Show All' to display all employees.")
            elif len(filtered_employees) == 0:
                st.warning("No employees found matching your search criteria.")
            else:
                # Display employee list - using checkbox instead of expander
                for i, employee in enumerate(filtered_employees):
                    # Find employee's index in original list
                    original_index = next((idx for idx, emp in enumerate(st.session_state.employees_data) 
                                        if emp['employee_id'] == employee['employee_id']), i)
                    
                    emp_id = employee['employee_id']
                    
                    # Use checkbox to control show/hide
                    show_details = st.checkbox(
                        f"{emp_id} - {employee['name']} | {employee['job_title']} | {employee['department']} | Skills: {employee.get('total_skills', 0)}", 
                        value=False, 
                        key=f"show_emp_{emp_id}"
                    )
                    
                    if show_details:
                        is_editing = st.session_state.get(f"edit_emp_{emp_id}", False)
                        
                        if is_editing:
                            with st.form(key=f"edit_form_emp_{emp_id}"):
                                st.write(f"**Editing Employee: {emp_id}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.text_input("Employee ID", value=emp_id, disabled=True, help="Employee ID cannot be changed")
                                    new_name = st.text_input("Name*", value=employee['name'])
                                    new_job_title = st.selectbox("Job Title*", [
                                        'Software Engineer', 'Senior Software Engineer', 'Frontend Developer', 'Backend Developer',
                                        'Full Stack Developer', 'Data Scientist', 'DevOps Engineer', 'Product Manager',
                                        'UI/UX Designer', 'Data Analyst', 'System Architect', 'Project Manager'],
                                        index=[
                                        'Software Engineer', 'Senior Software Engineer', 'Frontend Developer', 'Backend Developer',
                                        'Full Stack Developer', 'Data Scientist', 'DevOps Engineer', 'Product Manager',
                                        'UI/UX Designer', 'Data Analyst', 'System Architect', 'Project Manager'].index(employee['job_title'])
                                    )
                                    new_department = st.selectbox("Department*", ['Engineering', 'Data Science', 'Product', 'Design', 'DevOps', 'Other'],
                                        index=['Engineering', 'Data Science', 'Product', 'Design', 'DevOps', 'Other'].index(employee['department'])
                                    )
                                with col2:
                                    new_experience = st.number_input("Experience (Years)*", min_value=0, max_value=50, value=employee['experience_years'])
                                    new_performance = st.slider("Performance Rating*", 1.0, 5.0, value=employee['performance_rating'])
                                    new_email = st.text_input("Email*", value=employee['email'])
                                
                                # --- SKILL EDITING SECTION ---
                                st.subheader("Edit Skills & Proficiencies")
                                
                                # Parse existing skills
                                existing_skills = {}
                                if employee.get('skills'):
                                    for skill_item in employee['skills'].split('; '):
                                        if ':' in skill_item:
                                            skill, level = skill_item.split(':')
                                            existing_skills[skill.strip()] = int(level)
                                
                                edited_skills = {}
                                
                                for category, skills in self.SKILL_CATEGORIES.items():
                                    if category and skills:
                                        with st.expander(f"{category}"):
                                            # Use 5-column layout inside expander
                                            num_columns = 5
                                            columns = st.columns(num_columns)
                                            
                                            # Evenly distribute skills across 5 columns
                                            for idx, skill in enumerate(skills):
                                                if skill:
                                                    col_idx = idx % num_columns
                                                    with columns[col_idx]:
                                                        # Get current skill proficiency (if exists)
                                                        current_level = existing_skills.get(skill, 3)
                                                        
                                                        # Checkbox on top
                                                        if st.checkbox(f"{skill}", value=skill in existing_skills, key=f"edit_emp_skill_{emp_id}_{skill}"):
                                                            # Slider below - always shown but only saved when checked
                                                            proficiency = st.slider(
                                                                f"{skill} Level", 
                                                                1, 5, 
                                                                value=current_level,
                                                                key=f"edit_emp_prof_{emp_id}_{skill}"
                                                            )
                                                            edited_skills[skill] = proficiency
                                                        else:
                                                            # Also show slider when not checked, but don't save
                                                            st.slider(
                                                                f"{skill} Level", 
                                                                1, 5, 
                                                                value=1,
                                                                key=f"edit_emp_prof_disabled_{emp_id}_{skill}"
                                                            )
                                
                                # Calculate statistics (for saving)
                                new_skills_string = "; ".join([f"{skill}:{level}" for skill, level in edited_skills.items()])
                                new_total_skills = len(edited_skills)
                                new_avg_proficiency = round(sum(edited_skills.values()) / len(edited_skills), 1) if edited_skills else 0
        
                                save_clicked = st.form_submit_button("Save Changes")
                                cancel_clicked = st.form_submit_button("Cancel")
                                
                                if save_clicked:
                                    if not new_name or not new_email:
                                        st.error("Please fill in all required fields (marked with *)")
                                    else:
                                        # Update employee data
                                        st.session_state.employees_data[original_index]['name'] = new_name
                                        st.session_state.employees_data[original_index]['job_title'] = new_job_title
                                        st.session_state.employees_data[original_index]['department'] = new_department
                                        st.session_state.employees_data[original_index]['experience_years'] = new_experience
                                        st.session_state.employees_data[original_index]['performance_rating'] = new_performance
                                        st.session_state.employees_data[original_index]['email'] = new_email
                                        st.session_state.employees_data[original_index]['skills'] = new_skills_string
                                        st.session_state.employees_data[original_index]['total_skills'] = new_total_skills
                                        st.session_state.employees_data[original_index]['avg_proficiency'] = new_avg_proficiency
                                        
                                        st.session_state[f"edit_emp_{emp_id}"] = False
                                        self.add_notification(f"Employee updated: {new_name} ({emp_id})", "info")
                                        st.success(f"Employee {emp_id} updated successfully!")
                                        st.rerun()
                                
                                if cancel_clicked:
                                    st.session_state[f"edit_emp_{emp_id}"] = False
                                    st.rerun()
        
                        else:
                            # --- VIEW MODE ---
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Employee ID:** {emp_id}")
                                st.write(f"**Department:** {employee['department']}")
                                st.write(f"**Experience:** {employee['experience_years']} years")
                                st.write(f"**Performance:** {employee['performance_rating']}/5.0")
                            with col2:
                                st.write(f"**Email:** {employee['email']}")
                                st.write(f"**Total Skills:** {employee.get('total_skills', 0)}")
                                st.write(f"**Avg Proficiency:** {employee.get('avg_proficiency', 0)}")
                            
                            if employee['skills']:
                                st.write("**Skills:**")
                                skill_columns = st.columns(3)
                                skill_items = []
                                for skill_item in employee['skills'].split('; '):
                                    if ':' in skill_item:
                                        skill, level = skill_item.split(':')
                                        skill_items.append(f"- {skill}: {level}/5")
                                
                                items_per_column = (len(skill_items) + 2) // 3
                                for col_idx in range(3):
                                    start_idx = col_idx * items_per_column
                                    end_idx = min((col_idx + 1) * items_per_column, len(skill_items))
                                    with skill_columns[col_idx]:
                                        for item in skill_items[start_idx:end_idx]:
                                            st.write(item)
                            else:
                                st.write("**Skills:** No skills added")
                            
                            # --- Action Buttons ---
                            col_b1, col_b2, col_b_spacer = st.columns([1, 1, 3])
                            with col_b1:
                                if st.button("Edit Employee", key=f"edit_emp_btn_{emp_id}"):
                                    st.session_state[f"edit_emp_{emp_id}"] = True
                                    st.rerun()
                            with col_b2:
                                # Remove Employee Button
                                if st.button(f"Remove Employee", key=f"remove_emp_{emp_id}", type="primary"):
                                    assigned_to_projects = any(emp_id in proj.get('assigned_team', []) for proj in st.session_state.projects_data)
                                    assigned_to_tasks = any(task.get('assigned_to_id') == emp_id for task in st.session_state.tasks_data)
                                    
                                    if assigned_to_projects or assigned_to_tasks:
                                        st.error(f"Cannot remove {employee['name']}: Employee is assigned to projects or tasks. Please reassign first.")
                                    else:
                                        st.session_state.employees_data.pop(original_index)
                                        self.add_notification(f"Employee removed: {employee['name']}", "warning")
                                        st.rerun()
        
                    st.divider()
            
            st.divider()
    
        # Import/Export
        st.subheader("Data Import / Export")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Export Employee Data**")
            if st.session_state.employees_data:
                df_export = pd.DataFrame(st.session_state.employees_data)
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Employees as CSV",
                    data=csv_data,
                    file_name="employees_export.csv",
                    mime="text/csv",
                )
            else:
                st.info("No employee data to export.")
                
        with c2:
            st.write("**Import Employee Data (CSV)**")
            uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"], key="import_employee_csv")
            
            if uploaded_file is not None:
                try:
                    df_import = pd.read_csv(uploaded_file)
                    st.dataframe(df_import.head(), height=150)
                    
                    if st.button("Confirm Employee Import"):
                        existing_ids, _, _ = self.get_existing_ids()
                        new_employees_added = 0
                        
                        for record in df_import.to_dict('records'):
                            if 'employee_id' not in record:
                                st.error("CSV must contain an 'employee_id' column.")
                                break
                            
                            if record['employee_id'] not in existing_ids:
                                # Basic validation to ensure required keys exist
                                record.setdefault('name', 'N/A')
                                record.setdefault('job_title', 'N/A')
                                record.setdefault('department', 'Other')
                                record.setdefault('experience_years', 0)
                                record.setdefault('performance_rating', 3.0)
                                record.setdefault('email', 'N/A')
                                record.setdefault('skills', '')
                                record.setdefault('total_skills', 0)
                                record.setdefault('avg_proficiency', 0)
                                
                                st.session_state.employees_data.append(record)
                                existing_ids.append(record['employee_id'])
                                new_employees_added += 1
                        
                        st.success(f"Successfully imported {new_employees_added} new employees!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Failed to import file: {e}")

    def render_analytics_reports(self):
            """Render Analytics & Reports Tab"""
            st.header("📊 Analytics & Reports")
            
            if not st.session_state.projects_data and not st.session_state.tasks_data and not st.session_state.employees_data:
                st.info("No data available to generate analytics. Please add employees, projects, and tasks.")
                return
            
            # Part 1: Project Analysis
            st.subheader("📋 Project Analysis")
            
            if st.session_state.projects_data:
                # Project filters
                col1, col2 = st.columns(2)
                
                with col1:
                    project_status_filter = st.multiselect(
                        "Filter by Status:",
                        options=list(set(p['status'] for p in st.session_state.projects_data)),
                        default=list(set(p['status'] for p in st.session_state.projects_data)),
                        key="project_status_filter"
                    )
                
                with col2:
                    project_priority_filter = st.multiselect(
                        "Filter by Priority:",
                        options=list(set(p['Priority'] for p in st.session_state.projects_data)),
                        default=list(set(p['Priority'] for p in st.session_state.projects_data)),
                        key="project_priority_filter"
                    )
                
                # Filter projects
                filtered_projects = [p for p in st.session_state.projects_data 
                                   if p['status'] in project_status_filter 
                                   and p['Priority'] in project_priority_filter]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Project status distribution
                    st.markdown("**Status Distribution**")
                    if filtered_projects:
                        status_counts = pd.Series([p['status'] for p in filtered_projects]).value_counts().reset_index()
                        status_counts.columns = ['Status', 'Count']
                        
                        fig_status = px.pie(
                            status_counts, 
                            values='Count', 
                            names='Status',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hover_data=['Count']
                        )
                        fig_status.update_traces(
                            textposition='inside', 
                            textinfo='percent+label',
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                        )
                        st.plotly_chart(fig_status, use_container_width=True)
                    else:
                        st.info("No projects match the filters")
                
                with col2:
                    # Project priority distribution
                    st.markdown("**Priority Distribution**")
                    if filtered_projects:
                        priority_counts = pd.Series([p['Priority'] for p in filtered_projects]).value_counts().reset_index()
                        priority_counts.columns = ['Priority', 'Count']
                        
                        fig_priority = px.pie(
                            priority_counts, 
                            values='Count', 
                            names='Priority',
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            hover_data=['Count']
                        )
                        fig_priority.update_traces(
                            textposition='inside', 
                            textinfo='percent+label',
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                        )
                        st.plotly_chart(fig_priority, use_container_width=True)
                    else:
                        st.info("No projects match the filters")
                
                # Display project filter statistics
                if filtered_projects:
                    st.info(f"Showing {len(filtered_projects)} out of {len(st.session_state.projects_data)} projects")
            
            # Project Gantt chart
            if st.session_state.projects_data:
                st.markdown("**Project Timeline**")
                
                # Project Gantt chart filters - start time on far left
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Get date range for all projects
                    all_start_dates = [datetime.strptime(p['start_date'], '%Y-%m-%d') for p in st.session_state.projects_data]
                    all_end_dates = [datetime.strptime(p['deadline'], '%Y-%m-%d') for p in st.session_state.projects_data]
                    min_date = min(all_start_dates)
                    max_date = max(all_end_dates)
                    
                    project_start_from = st.date_input(
                        "Start Date From:",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="project_start_from"
                    )
                
                with col2:
                    project_start_to = st.date_input(
                        "Start Date To:",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        key="project_start_to"
                    )
                
                with col3:
                    gantt_status_filter = st.multiselect(
                        "Filter by Status:",
                        options=list(set(p['status'] for p in st.session_state.projects_data)),
                        default=list(set(p['status'] for p in st.session_state.projects_data)),
                        key="project_status_gantt_filter"
                    )
                
                with col4:
                    gantt_priority_filter = st.multiselect(
                        "Filter by Priority:",
                        options=list(set(p['Priority'] for p in st.session_state.projects_data)),
                        default=list(set(p['Priority'] for p in st.session_state.projects_data)),
                        key="project_priority_gantt_filter"
                    )
                
                # Filter projects
                filtered_gantt_projects = []
                for project in st.session_state.projects_data:
                    project_start = datetime.strptime(project['start_date'], '%Y-%m-%d')
                    project_status = project['status']
                    project_priority = project['Priority']
                    
                    if (project_start.date() >= project_start_from and
                        project_start.date() <= project_start_to and
                        project_status in gantt_status_filter and
                        project_priority in gantt_priority_filter):
                        filtered_gantt_projects.append(project)
                
                if filtered_gantt_projects:
                    # Create project Gantt chart data
                    project_gantt_data = []
                    for project in filtered_gantt_projects:
                        progress = self.calculate_project_progress(project['project_id'])
                        start_date = datetime.strptime(project['start_date'], '%Y-%m-%d')
                        end_date = datetime.strptime(project['deadline'], '%Y-%m-%d')
                        duration_days = (end_date - start_date).days
                        
                        project_gantt_data.append({
                            'Task': f"{project['project_id']} - {project['project_name']}",
                            'Start': project['start_date'],
                            'Finish': project['deadline'],
                            'Status': project['status'],
                            'Priority': project['Priority'],
                            'Progress': progress,
                            'Duration_Days': duration_days,
                            'Team_Size': len(project.get('assigned_team', [])),
                            'Complexity': project['Complexity_Score']
                        })
                    
                    project_gantt_df = pd.DataFrame(project_gantt_data)
                    
                    # Create project Gantt chart
                    fig_project_gantt = px.timeline(
                        project_gantt_df,
                        x_start="Start",
                        x_end="Finish",
                        y="Task",
                        color="Status",
                        hover_data=["Priority", "Progress", "Duration_Days", "Team_Size", "Complexity"],
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    
                    fig_project_gantt.update_traces(
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Duration: %{customdata[2]} days<br>"
                            "Progress: %{customdata[1]}%<br>"
                            "Priority: %{customdata[0]}<br>"
                            "Team Size: %{customdata[3]}<br>"
                            "Complexity: %{customdata[4]}/10<br>"
                            "Start: %{x|%Y-%m-%d}<br>"
                            "End: %{x_end|%Y-%m-%d}"
                        )
                    )
                    
                    fig_project_gantt.update_yaxes(autorange="reversed")
                    fig_project_gantt.update_layout(
                        height=400,
                        xaxis_title="Timeline",
                        yaxis_title="Projects",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_project_gantt, use_container_width=True)
                    
                    # Display time range statistics
                    st.info(f"Showing {len(filtered_gantt_projects)} projects with start dates from {project_start_from} to {project_start_to}")
                else:
                    st.warning("No projects match the selected filters.")
            
            # Part 2: Task Analysis
            st.subheader("✅ Task Analysis")
            
            if st.session_state.tasks_data:
                # Task filters
                col1, col2 = st.columns(2)
                
                with col1:
                    task_status_filter = st.multiselect(
                        "Filter by Status:",
                        options=list(set(t['status'] for t in st.session_state.tasks_data)),
                        default=list(set(t['status'] for t in st.session_state.tasks_data)),
                        key="task_status_filter"
                    )
                
                with col2:
                    task_category_filter = st.multiselect(
                        "Filter by Category:",
                        options=list(set(t['category'] for t in st.session_state.tasks_data)),
                        default=list(set(t['category'] for t in st.session_state.tasks_data)),
                        key="task_category_filter"
                    )
                
                # Filter tasks
                filtered_tasks = [t for t in st.session_state.tasks_data 
                                 if t['status'] in task_status_filter 
                                 and t['category'] in task_category_filter]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Task status distribution
                    st.markdown("**Status Distribution**")
                    if filtered_tasks:
                        task_status_counts = pd.Series([t['status'] for t in filtered_tasks]).value_counts().reset_index()
                        task_status_counts.columns = ['Status', 'Count']
                        
                        fig_task_status = px.pie(
                            task_status_counts,
                            values='Count',
                            names='Status',
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hover_data=['Count']
                        )
                        fig_task_status.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                        )
                        st.plotly_chart(fig_task_status, use_container_width=True)
                    else:
                        st.info("No tasks match the filters")
                
                with col2:
                    # Task priority distribution
                    st.markdown("**Priority Distribution**")
                    if filtered_tasks:
                        task_priority_data = []
                        for task in filtered_tasks:
                            # Assign priority to tasks (based on complexity and duration)
                            complexity = task.get('complexity', 3)
                            duration = task.get('estimated_duration', 7)
                            
                            # Simple priority calculation logic
                            if complexity >= 4 or duration >= 14:
                                priority = 'High'
                            elif complexity >= 3 or duration >= 7:
                                priority = 'Medium'
                            else:
                                priority = 'Low'
                            
                            task_priority_data.append({
                                'Priority': priority,
                                'Status': task['status']
                            })
                        
                        task_priority_df = pd.DataFrame(task_priority_data)
                        priority_counts = task_priority_df['Priority'].value_counts().reset_index()
                        priority_counts.columns = ['Priority', 'Count']
                        
                        fig_task_priority = px.pie(
                            priority_counts,
                            values='Count',
                            names='Priority',
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            hover_data=['Count']
                        )
                        fig_task_priority.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                        )
                        st.plotly_chart(fig_task_priority, use_container_width=True)
                    else:
                        st.info("No tasks match the filters")
                
                # Display task filter statistics
                if filtered_tasks:
                    st.info(f"Showing {len(filtered_tasks)} out of {len(st.session_state.tasks_data)} tasks")
            
            # Task Gantt chart
            if st.session_state.tasks_data:
                st.markdown("**Task Timeline**")
                
                # Task Gantt chart filters - start time on far left
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Task start time filter
                    task_start_dates = []
                    for task in st.session_state.tasks_data:
                        created_date = task.get('created_date', datetime.now().strftime('%Y-%m-%d'))
                        if ' ' in created_date:
                            start_date = datetime.strptime(created_date.split()[0], '%Y-%m-%d')
                        else:
                            start_date = datetime.strptime(created_date, '%Y-%m-%d')
                        task_start_dates.append(start_date)
                    
                    if task_start_dates:
                        min_task_start = min(task_start_dates)
                        max_task_start = max(task_start_dates)
                        
                        task_start_from = st.date_input(
                            "Start Date From:",
                            value=min_task_start,
                            min_value=min_task_start,
                            max_value=max_task_start,
                            key="task_start_from"
                        )
                    else:
                        task_start_from = datetime.now().date()
                
                with col2:
                    if task_start_dates:
                        task_start_to = st.date_input(
                            "Start Date To:",
                            value=max_task_start,
                            min_value=min_task_start,
                            max_value=max_task_start,
                            key="task_start_to"
                        )
                    else:
                        task_start_to = datetime.now().date()
                
                with col3:
                    # Filter tasks by project
                    project_options = ["All Projects"] + [p['project_id'] for p in st.session_state.projects_data]
                    selected_project = st.selectbox(
                        "Filter by Project:",
                        options=project_options,
                        key="task_project_filter"
                    )
                
                with col4:
                    task_gantt_status_filter = st.multiselect(
                        "Filter by Status:",
                        options=list(set(t['status'] for t in st.session_state.tasks_data)),
                        default=list(set(t['status'] for t in st.session_state.tasks_data)),
                        key="task_status_gantt_filter"
                    )
                
                # Filter tasks
                filtered_gantt_tasks = []
                for task in st.session_state.tasks_data:
                    task_status = task['status']
                    task_project = task.get('project_id')
                    
                    # Get task start date
                    created_date = task.get('created_date', datetime.now().strftime('%Y-%m-%d'))
                    if ' ' in created_date:
                        task_start_date = datetime.strptime(created_date.split()[0], '%Y-%m-%d')
                    else:
                        task_start_date = datetime.strptime(created_date, '%Y-%m-%d')
                    
                    # Apply filter conditions
                    project_match = (selected_project == "All Projects" or task_project == selected_project)
                    status_match = task_status in task_gantt_status_filter
                    start_date_match = (task_start_date.date() >= task_start_from and 
                                      task_start_date.date() <= task_start_to)
                    
                    if project_match and status_match and start_date_match:
                        filtered_gantt_tasks.append(task)
                
                if filtered_gantt_tasks:
                    # Create task Gantt chart data
                    task_gantt_data = []
                    for task in filtered_gantt_tasks:
                        task_deadline = task.get('deadline')
                        created_date = task.get('created_date', datetime.now().strftime('%Y-%m-%d'))
                        
                        # Calculate start date (creation date or duration-based estimate)
                        start_date = datetime.strptime(created_date.split()[0], '%Y-%m-%d') if ' ' in created_date else datetime.strptime(created_date, '%Y-%m-%d')
                        
                        if task_deadline:
                            end_date = datetime.strptime(task_deadline, '%Y-%m-%d')
                            duration_days = (end_date - start_date).days
                        else:
                            end_date = start_date + timedelta(days=task.get('estimated_duration', 7))
                            duration_days = task.get('estimated_duration', 7)
                        
                        task_gantt_data.append({
                            'Task': f"{task['task_id']} - {task['task_name']}",
                            'Start': start_date.strftime('%Y-%m-%d'),
                            'Finish': end_date.strftime('%Y-%m-%d'),
                            'Status': task['status'],
                            'Category': task.get('category', 'Other'),
                            'Project': task.get('project_id', 'Unknown'),
                            'Duration_Days': duration_days,
                            'Complexity': task.get('complexity', 3),
                            'Assigned_To': task.get('assigned_to_name', 'Unassigned')
                        })
                    
                    task_gantt_df = pd.DataFrame(task_gantt_data)
                    
                    # Create task Gantt chart
                    fig_task_gantt = px.timeline(
                        task_gantt_df,
                        x_start="Start",
                        x_end="Finish",
                        y="Task",
                        color="Status",
                        hover_data=["Category", "Project", "Duration_Days", "Complexity", "Assigned_To"],
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig_task_gantt.update_traces(
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Project: %{customdata[1]}<br>"
                            "Category: %{customdata[0]}<br>"
                            "Duration: %{customdata[2]} days<br>"
                            "Complexity: %{customdata[3]}/5<br>"
                            "Assigned To: %{customdata[4]}<br>"
                            "Start: %{x|%Y-%m-%d}<br>"
                            "End: %{x_end|%Y-%m-%d}"
                        )
                    )
                    
                    fig_task_gantt.update_yaxes(autorange="reversed")
                    fig_task_gantt.update_layout(
                        height=400,
                        xaxis_title="Timeline",
                        yaxis_title="Tasks",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_task_gantt, use_container_width=True)
                    
                    # Display time range statistics
                    st.info(f"Showing {len(filtered_gantt_tasks)} tasks with start dates from {task_start_from} to {task_start_to}")
                else:
                    st.warning("No tasks match the selected filters.")
            
                # Part 3: Team & Performance Analysis
                st.subheader("👥 Team & Performance Analysis")
                
                if st.session_state.employees_data:
                    # Add department filter
                    col1, = st.columns(1)
                    
                    with col1:
                        dept_filter = st.multiselect(
                            "Filter by Department:",
                            options=list(set(e['department'] for e in st.session_state.employees_data)),
                            default=list(set(e['department'] for e in st.session_state.employees_data)),
                            key="employee_dept_filter"
                        )
                    
                    # Initialize variables
                    filtered_employees = []
                    filtered_employee_df = pd.DataFrame()
                    
                    # Check if departments are selected
                    if not dept_filter:
                        st.warning("Please select at least one department to view employee data.")
                    else:
                        # Only show table when departments are selected
                        st.markdown("**Employee Overview**")
                        employee_df = pd.DataFrame(st.session_state.employees_data)
                        
                        # Filter employees
                        filtered_employees = [e for e in st.session_state.employees_data 
                                            if e['department'] in dept_filter]
                        
                        filtered_employee_df = pd.DataFrame(filtered_employees)
                    
                    # Handle data display uniformly outside conditional statement
                    if not filtered_employee_df.empty:
                        # Display employee data table
                        st.dataframe(
                            filtered_employee_df[['employee_id', 'name', 'job_title', 'department', 
                                               'experience_years', 'performance_rating', 'total_skills', 'avg_proficiency']],
                            use_container_width=True,
                            height=300
                        )
                    else:
                        # Handle empty data case
                        st.info("No employee data to display.")
                    
                    # Team analysis charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Department distribution pie chart
                        st.markdown("**Department Distribution**")
                        if filtered_employees:  # This variable is now always defined
                            dept_counts = pd.Series([e['department'] for e in filtered_employees]).value_counts().reset_index()
                            dept_counts.columns = ['Department', 'Count']
                            
                            fig_dept = px.pie(
                                dept_counts,
                                values='Count',
                                names='Department',
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            fig_dept.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                hovertemplate="<b>%{label}</b><br>Employees: %{value}<extra></extra>"
                            )
                            st.plotly_chart(fig_dept, use_container_width=True)
                        else:
                            st.info("No employees match the filters")
                        
                        # Experience distribution histogram (with edge lines)
                        if filtered_employees:  # This variable is now always defined
                            st.markdown("**Experience Distribution**")
                            fig_exp_hist = px.histogram(
                                filtered_employee_df,
                                x='experience_years',
                                nbins=10,
                                color_discrete_sequence=['#636EFA'],
                                opacity=0.8
                            )
                            fig_exp_hist.update_traces(
                                marker_line_color='black',  # Add black edge lines
                                marker_line_width=1.5,      # Edge line width
                                opacity=0.8                 # Slightly transparent to show edges
                            )
                            fig_exp_hist.update_layout(
                                xaxis_title="Experience (Years)",
                                yaxis_title="Number of Employees",
                                bargap=0.1  # Set gap between bars
                            )
                            st.plotly_chart(fig_exp_hist, use_container_width=True)
                    
                    with col2:
                        # Performance rating box plot
                        st.markdown("**Performance Rating by Department**")
                        if filtered_employees:  # This variable is now always defined
                            fig_perf_box = px.box(
                                filtered_employee_df,
                                x='department',
                                y='performance_rating',
                                color='department',
                                points="all",
                                hover_data=['name', 'job_title']
                            )
                            fig_perf_box.update_layout(
                                xaxis_title="Department",
                                yaxis_title="Performance Rating (1-5)"
                            )
                            st.plotly_chart(fig_perf_box, use_container_width=True)
                        else:
                            st.info("No employees match the filters")
                        
                        # Skills vs performance relationship
                        if filtered_employees:  # This variable is now always defined
                            st.markdown("**Skills vs Performance**")
                            fig_skills_perf = px.scatter(
                                filtered_employee_df,
                                x='total_skills',
                                y='performance_rating',
                                size='experience_years',
                                color='department',
                                hover_data=['name', 'job_title', 'avg_proficiency'],
                                size_max=20
                            )
                            fig_skills_perf.update_layout(
                                xaxis_title="Number of Skills",
                                yaxis_title="Performance Rating"
                            )
                            st.plotly_chart(fig_skills_perf, use_container_width=True)
                    
                    # Team statistics summary
                    if filtered_employees:
                        st.markdown("**Team Statistics Summary**")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            total_employees = len(filtered_employees)
                            st.metric("Total Employees", total_employees)
                        
                        with stat_col2:
                            avg_performance = filtered_employee_df['performance_rating'].mean()
                            st.metric("Avg Performance", f"{avg_performance:.1f}/5.0")
                        
                        with stat_col3:
                            avg_experience = filtered_employee_df['experience_years'].mean()
                            st.metric("Avg Experience", f"{avg_experience:.1f} years")
                        
                        with stat_col4:
                            avg_skills = filtered_employee_df['total_skills'].mean()
                            st.metric("Avg Skills", f"{avg_skills:.1f}")
    
                    # --- NEW SECTION: Employee Task Workload ---
                    st.markdown("---")
                    st.subheader("🛠️ Employee Workload & Task Progress")
                    
                    if st.session_state.tasks_data and filtered_employees:
                        # 1. Filters for this specific section (Status & Employee)
                        col_f1, col_f2 = st.columns(2)
                        
                        with col_f1:
                            # Workload Status Filter with "Select All" toggle
                            available_statuses = sorted(list(set(t['status'] for t in st.session_state.tasks_data)))
                            
                            # Add Checkbox for Select All
                            select_all_status = st.checkbox("Select All Statuses", value=True, key="chk_all_statuses_workload")
                            
                            if select_all_status:
                                workload_status_filter = available_statuses
                                # Show disabled multiselect for visual confirmation
                                st.multiselect(
                                    "Filter Workload by Task Status:",
                                    options=available_statuses,
                                    default=available_statuses,
                                    disabled=True,
                                    key="disabled_status_filter_workload"
                                )
                            else:
                                workload_status_filter = st.multiselect(
                                    "Filter Workload by Task Status:",
                                    options=available_statuses,
                                    default=available_statuses,
                                    key="emp_workload_status_filter"
                                )
    
                        with col_f2:
                             # Employee Name Filter with "Select All" toggle
                            emp_names_list = [e['name'] for e in filtered_employees]
                            
                            # Add Checkbox for Select All
                            select_all_emps = st.checkbox("Select All Employees", value=True, key="chk_all_emps_workload")
                            
                            if select_all_emps:
                                selected_emp_names = emp_names_list
                                # Show disabled multiselect for visual confirmation
                                st.multiselect(
                                    "Focus on specific employees:",
                                    options=emp_names_list,
                                    default=emp_names_list,
                                    disabled=True,
                                    key="disabled_emp_filter_workload"
                                )
                            else:
                                selected_emp_names = st.multiselect(
                                    "Focus on specific employees:",
                                    options=emp_names_list,
                                    default=emp_names_list,
                                    key="emp_workload_name_filter"
                                )
                        
                        # 2. Data Processing
                        # Filter tasks that belong to the selected department/employees AND match status filter
                        workload_data = []
                        
                        # Create a map of ID -> Name for easier lookup
                        emp_id_to_name = {e['employee_id']: e['name'] for e in filtered_employees}
                        
                        for task in st.session_state.tasks_data:
                            assignee_id = task.get('assigned_to_id')
                            status = task.get('status', 'Unassigned')
                            
                            # Conditions:
                            # 1. Task is assigned to an employee in the current department filter
                            # 2. That employee is selected in the name filter
                            # 3. Status matches status filter
                            if (assignee_id in emp_id_to_name and 
                                emp_id_to_name[assignee_id] in selected_emp_names and
                                status in workload_status_filter):
                                
                                workload_data.append({
                                    'Employee': emp_id_to_name[assignee_id],
                                    'Task': task['task_name'],
                                    'Status': status,
                                    'Priority': task.get('complexity', 'Medium'), # using complexity as proxy if priority missing
                                    'Count': 1
                                })
    
                        # 3. Visualization
                        if workload_data:
                            workload_df = pd.DataFrame(workload_data)
                            
                            # Stacked Bar Chart: Task Count per Employee colored by Status
                            st.markdown("**Task Count & Status per Employee**")
                            
                            # Grouping for the chart
                            chart_df = workload_df.groupby(['Employee', 'Status']).size().reset_index(name='Task Count')
                            
                            fig_workload = px.bar(
                                chart_df,
                                x='Employee',
                                y='Task Count',
                                color='Status',
                                title="Workload Distribution",
                                text_auto=True,
                                color_discrete_sequence=px.colors.qualitative.Prism
                            )
                            
                            fig_workload.update_layout(
                                xaxis_title="Employee",
                                yaxis_title="Number of Tasks",
                                legend_title="Task Status"
                            )
                            
                            st.plotly_chart(fig_workload, use_container_width=True)
                            
                            # 4. Detailed Metrics Table
                            st.markdown("**Detailed Progress Metrics**")
                            
                            # Calculate metrics per employee
                            metrics_data = []
                            for name in selected_emp_names:
                                emp_tasks = [t for t in st.session_state.tasks_data 
                                            if t.get('assigned_to_name') == name] # Get ALL tasks for this person, regardless of visual filter
                                
                                if not emp_tasks:
                                    continue
                                    
                                total = len(emp_tasks)
                                completed = len([t for t in emp_tasks if t['status'] == 'Completed'])
                                in_progress = len([t for t in emp_tasks if t['status'] == 'In Progress'])
                                
                                completion_rate = (completed / total * 100) if total > 0 else 0
                                
                                metrics_data.append({
                                    'Employee': name,
                                    'Total Tasks': total,
                                    'Active (In Progress)': in_progress,
                                    'Completed': completed,
                                    'Completion Rate': f"{completion_rate:.1f}%"
                                })
                                
                            if metrics_data:
                                metrics_df = pd.DataFrame(metrics_data)
                                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                
                        else:
                            st.info("No tasks found matching the selected filters for these employees.")
                    
                    elif not st.session_state.tasks_data:
                        st.info("No tasks available in the system to analyze.")
                    elif not filtered_employees:
                        st.info("No employees selected.")

    def render_documents_issues(self):
        """Render Documents & Issues Tab"""
        st.header("📁 Documents & Issues")
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("Project Documents")
            
            # Document upload form
            with st.form("doc_upload_form"):
                if st.session_state.projects_data:
                    project_options = [f"{p['project_id']} - {p['project_name']}" for p in st.session_state.projects_data]
                    selected_project = st.selectbox("Select Project*", project_options)
                    project_id = selected_project.split(" - ")[0]
                else:
                    project_id = st.text_input("Project ID*", placeholder="PROJ-001")
            
                doc_title = st.text_input("Document Title*", placeholder="Initial Scope")
                doc_version = st.text_input("Version*", placeholder="v1.0")
                uploaded_file = st.file_uploader("Upload File*")
                
                submitted = st.form_submit_button("Upload Document")
                
                if submitted:
                    if project_id and doc_title and doc_version and uploaded_file is not None:
                        doc_id = str(uuid.uuid4())[:8]
                        # Save file content for download
                        doc_data = {
                            'doc_id': doc_id,
                            'project_id': project_id,
                            'title': doc_title,
                            'version': doc_version,
                            'file_name': uploaded_file.name,
                            'file_content': uploaded_file.getvalue(),
                            'uploaded_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                        }
                        st.session_state.documents_data.append(doc_data)
                        # Add notification
                        self.add_notification(f"Document '{doc_title}' uploaded for project {project_id}", "info", project_id)
                        st.success(f"Document '{doc_title}' uploaded for project {project_id}.")
                    else:
                        st.error("Please fill in all fields and upload a file.")
            
            st.divider()
            st.subheader("Uploaded Documents")
            if st.session_state.documents_data:
                for i, doc in enumerate(st.session_state.documents_data):
                    with st.container(border=True):
                        # get file info
                        doc_title = doc.get('title')
                        doc_version = doc.get('version', 'N/A')
                        project_ref = doc.get('project_id')
                        file_name = doc.get('file_name', 'No file attached')
                        uploaded_at = doc.get('uploaded_at')
                        
                        # show file info
                        st.write(f"**{doc_title} ({doc_version})** - Project: {project_ref}")
                        st.write(f"*File:* {file_name} | *Uploaded:* {uploaded_at}")
                        
                        # downlaod button
                        if 'file_content' in doc and doc['file_content']:
                            st.download_button(
                                label=f"Download {file_name}",
                                data=doc['file_content'],
                                file_name=file_name,
                                key=f"download_doc_{i}"
                            )
                        else:
                            st.info("No file content available for download")
                        
                        # Delete button
                        if st.button("Delete Document", key=f"delete_doc_{i}"):
                            doc_title = st.session_state.documents_data[i].get('title')
                            project_id = st.session_state.documents_data[i].get('project_id')
                            st.session_state.documents_data.pop(i)
                            # Add notification
                            self.add_notification(f"Document '{doc_title}' deleted from project {project_id}", "warning", project_id)
                            st.rerun()
            
            else:
                st.info("No documents uploaded yet.")
    
        with col2:
            st.subheader("Project Issues")
            
            # Issue creation form
            with st.form("issue_form"):
                if st.session_state.projects_data:
                    project_options = [f"{p['project_id']} - {p['project_name']}" for p in st.session_state.projects_data]
                    selected_project = st.selectbox("Select Project*", project_options, key="issue_proj")
                    project_id = selected_project.split(" - ")[0]
                else:
                    project_id = st.text_input("Project ID*", placeholder="PROJ-001", key="issue_proj_text")
    
                issue_title = st.text_input("Issue Title*", placeholder="Login button not working on mobile")
                issue_priority = st.selectbox("Priority*", ['Low', 'Medium', 'High'])
                issue_description = st.text_area("Description*", placeholder="Describe the issue in detail...")
                
                submitted = st.form_submit_button("Submit Issue")
                
                if submitted:
                    if project_id and issue_title and issue_description:
                        issue_id = str(uuid.uuid4())[:8]
                        issue_data = {
                            'issue_id': issue_id,
                            'project_id': project_id,
                            'title': issue_title,
                            'priority': issue_priority,
                            'description': issue_description,
                            'status': 'Open',
                            'reported_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'comments': []
                        }
                        st.session_state.issues_data.append(issue_data)
                        # Add notification
                        self.add_notification(f"New issue reported: '{issue_title}' for project {project_id}", "warning", project_id)
                        st.success(f"Issue '{issue_title}' submitted for project {project_id}.")
                    else:
                        st.error("Please fill in all fields.")
    
            st.divider()
            st.subheader("Logged Issues")
            if st.session_state.issues_data:
                for i, issue in enumerate(st.session_state.issues_data):
                    issue_id = issue['issue_id']
                    with st.expander(f"[{issue['status']}] {issue['title']} (Priority: {issue['priority']})"):
                        st.write(f"**Project:** {issue['project_id']} | **Reported:** {issue['reported_at']}")
                        st.write(f"**Description:** {issue['description']}")
                        
                        # Update status
                        status_options = ['Open', 'In Progress', 'Blocked', 'Solved', 'Closed']
                        current_status_index = status_options.index(issue['status'])
                        new_status = st.selectbox("Update Status", status_options, index=current_status_index, key=f"issue_status_{issue_id}")
                        
                        if new_status != issue['status']:
                            st.session_state.issues_data[i]['status'] = new_status
                            # Add notification
                            self.add_notification(f"Issue '{issue['title']}' status updated to {new_status}", "info", issue['project_id'])
                            st.info(f"Issue status updated to {new_status}")
                            st.rerun()
                        
                        st.divider()
                        
                        # Comments section
                        st.subheader("Comments")
                        if not issue['comments']:
                            st.info("No comments yet.")
                        else:
                            for comment in issue['comments']:
                                st.write(f"**{comment['user']}** ({comment['time']}):")
                                st.write(f"> {comment['text']}")
                        
                        with st.form(key=f"comment_form_{issue_id}"):
                            comment_text = st.text_area("Add a comment:", key=f"comment_text_{issue_id}")
                            if st.form_submit_button("Post Comment"):
                                if comment_text:
                                    st.session_state.issues_data[i]['comments'].append({
                                        'user': 'Admin',
                                        'text': comment_text,
                                        'time': datetime.now().strftime('%Y-%m-%d %H:%M')
                                    })
                                    # Add notification
                                    self.add_notification(f"Comment added to issue: '{issue['title']}'", "info", issue['project_id'])
                                    st.rerun()
                                else:
                                    st.error("Comment cannot be empty.")
                        
                        # Delete issue button
                        if st.button("Delete Issue", key=f"delete_issue_{issue_id}"):
                            issue_title = st.session_state.issues_data[i].get('title')
                            project_id = st.session_state.issues_data[i].get('project_id')
                            st.session_state.issues_data.pop(i)
                            # Add notification
                            self.add_notification(f"Issue '{issue_title}' deleted from project {project_id}", "warning", project_id)
                            st.rerun()
            else:
                st.info("No issues logged yet.")

    def generate_sample_data(self):
        """Generate comprehensive sample data for demonstration"""
        
        # Clear existing data first
        st.session_state.employees_data = []
        st.session_state.projects_data = []
        st.session_state.tasks_data = []
        st.session_state.documents_data = []
        st.session_state.issues_data = []
        st.session_state.notifications_data = []
        
        # Sample Employees
        sample_employees = [
            {
                'employee_id': 'EMP-001',
                'name': 'Alice Johnson',
                'job_title': 'Senior Software Engineer',
                'experience_years': 8,
                'performance_rating': 4.7,
                'email': 'alice.johnson@company.com',
                'department': 'Engineering',
                'skills': 'Python:5; JavaScript:4; React:4; AWS:4; Docker:4; Machine Learning:4',
                'total_skills': 6,
                'avg_proficiency': 4.2
            },
            {
                'employee_id': 'EMP-002',
                'name': 'Bob Chen',
                'job_title': 'Frontend Developer',
                'experience_years': 4,
                'performance_rating': 4.3,
                'email': 'bob.chen@company.com',
                'department': 'Engineering',
                'skills': 'JavaScript:5; React:5; Vue.js:4; HTML:5; CSS:5; TypeScript:4',
                'total_skills': 6,
                'avg_proficiency': 4.7
            },
            {
                'employee_id': 'EMP-003',
                'name': 'Carol Davis',
                'job_title': 'Data Scientist',
                'experience_years': 6,
                'performance_rating': 4.5,
                'email': 'carol.davis@company.com',
                'department': 'Data Science',
                'skills': 'Python:5; Machine Learning:5; TensorFlow:4; Pandas:5; SQL:4; Data Analysis:5',
                'total_skills': 6,
                'avg_proficiency': 4.7
            },
            {
                'employee_id': 'EMP-004',
                'name': 'David Wilson',
                'job_title': 'DevOps Engineer',
                'experience_years': 5,
                'performance_rating': 4.4,
                'email': 'david.wilson@company.com',
                'department': 'DevOps',
                'skills': 'Docker:5; Kubernetes:4; AWS:5; Linux:5; CI/CD:4; Bash:5',
                'total_skills': 6,
                'avg_proficiency': 4.7
            },
            {
                'employee_id': 'EMP-005',
                'name': 'Eva Martinez',
                'job_title': 'Project Manager',
                'experience_years': 7,
                'performance_rating': 4.6,
                'email': 'eva.martinez@company.com',
                'department': 'Product',
                'skills': 'Project Management:5; Agile:5; Scrum:4; Communication:5; Leadership:4',
                'total_skills': 5,
                'avg_proficiency': 4.6
            },
            {
                'employee_id': 'EMP-006',
                'name': 'Frank Lee',
                'job_title': 'Backend Developer',
                'experience_years': 3,
                'performance_rating': 4.2,
                'email': 'frank.lee@company.com',
                'department': 'Engineering',
                'skills': 'Java:4; Spring:4; SQL:4; MongoDB:3; Docker:3; REST API:4',
                'total_skills': 6,
                'avg_proficiency': 3.7
            },
            {
                'employee_id': 'EMP-007',
                'name': 'Grace Kim',
                'job_title': 'UI/UX Designer',
                'experience_years': 4,
                'performance_rating': 4.4,
                'email': 'grace.kim@company.com',
                'department': 'Design',
                'skills': 'Figma:5; Adobe XD:4; User Research:4; Prototyping:5; Design System:4',
                'total_skills': 5,
                'avg_proficiency': 4.4
            },
            {
                'employee_id': 'EMP-008',
                'name': 'Henry Brown',
                'job_title': 'Full Stack Developer',
                'experience_years': 5,
                'performance_rating': 4.5,
                'email': 'henry.brown@company.com',
                'department': 'Engineering',
                'skills': 'Python:4; Django:4; React:4; PostgreSQL:4; AWS:3; Docker:4',
                'total_skills': 6,
                'avg_proficiency': 3.8
            }
        ]
        
        # Sample Projects
        sample_projects = [
            {
                'project_id': 'PROJ-001',
                'project_name': 'Quantum Analytics Platform',
                'description': 'Develop a comprehensive data analytics platform for financial services with real-time dashboard and machine learning capabilities.',
                'start_date': '2025-01-15',
                'deadline': '2025-06-30',
                'status': 'In Progress',
                'manager_id': 'EMP-005',
                'Complexity_Score': 8,
                'Priority': 'High',
                'Priority_Score': 3,
                'required_skillsets': 'Python; Machine Learning; React; AWS; Data Analysis; Docker',
                'created_date': '2025-01-10 09:00:00',
                'assigned_team': ['EMP-001', 'EMP-003', 'EMP-002']
            },
            {
                'project_id': 'PROJ-002',
                'project_name': 'Nexus E-commerce Redesign',
                'description': 'Redesign and modernize the e-commerce platform with improved user experience and mobile-first design.',
                'start_date': '2025-02-01',
                'deadline': '2025-04-30',
                'status': 'In Progress',
                'manager_id': 'EMP-005',
                'Complexity_Score': 6,
                'Priority': 'Medium',
                'Priority_Score': 2,
                'required_skillsets': 'React; JavaScript; Figma; CSS; User Research',
                'created_date': '2025-01-25 14:30:00',
                'assigned_team': ['EMP-002', 'EMP-007']
            },
            {
                'project_id': 'PROJ-003',
                'project_name': 'Cloud Infrastructure Migration',
                'description': 'Migrate legacy on-premise infrastructure to cloud-native architecture with containerization and auto-scaling.',
                'start_date': '2025-03-01',
                'deadline': '2025-08-31',
                'status': 'Not Started',
                'manager_id': 'EMP-004',
                'Complexity_Score': 9,
                'Priority': 'High',
                'Priority_Score': 3,
                'required_skillsets': 'Docker; Kubernetes; AWS; Linux; CI/CD',
                'created_date': '2025-02-15 11:00:00',
                'assigned_team': ['EMP-004']
            },
            {
                'project_id': 'PROJ-004',
                'project_name': 'Customer Portal Enhancement',
                'description': 'Enhance customer self-service portal with new features and improved performance.',
                'start_date': '2025-01-10',
                'deadline': '2025-03-31',
                'status': 'Completed',
                'manager_id': 'EMP-005',
                'Complexity_Score': 5,
                'Priority': 'Medium',
                'Priority_Score': 2,
                'required_skillsets': 'Java; Spring; React; SQL',
                'created_date': '2025-01-05 10:00:00',
                'assigned_team': ['EMP-006', 'EMP-008']
            }
        ]
        
        # Sample Tasks - EXPANDED to show multiple statuses per employee
        sample_tasks = [
            # --- Alice Johnson (EMP-001) - Mixed Statuses ---
            {
                'task_id': 'TASK-1001',
                'project_id': 'PROJ-001',
                'task_name': 'Design Data Pipeline',
                'task_description': 'Design and implement the core data processing pipeline for real-time analytics.',
                'complexity': 4,
                'category': 'Backend',
                'estimated_duration': 21,
                'estimated_budget': 8500,
                'start_date': '2025-01-20',
                'deadline': '2025-03-15',
                'status': 'Completed', # Completed task
                'assigned_to_id': 'EMP-001',
                'assigned_to_name': 'Alice Johnson',
                'required_skillsets': 'Python; AWS; Data Analysis',
                'created_date': '2025-01-15 09:00:00'
            },
            {
                'task_id': 'TASK-1004',
                'project_id': 'PROJ-001',
                'task_name': 'Optimize Query Performance',
                'task_description': 'Analyze and optimize slow database queries for the analytics engine.',
                'complexity': 5,
                'category': 'Database',
                'estimated_duration': 10,
                'estimated_budget': 3000,
                'start_date': '2025-03-16',
                'deadline': '2025-03-30',
                'status': 'In Progress', # In Progress task
                'assigned_to_id': 'EMP-001',
                'assigned_to_name': 'Alice Johnson',
                'required_skillsets': 'SQL; PostgreSQL; Performance Tuning',
                'created_date': '2025-02-15 09:00:00'
            },
            {
                'task_id': 'TASK-1005',
                'project_id': 'PROJ-001',
                'task_name': 'Security Audit Integration',
                'task_description': 'Integrate security compliance checks into the data pipeline.',
                'complexity': 4,
                'category': 'Security',
                'estimated_duration': 7,
                'estimated_budget': 2500,
                'start_date': '2025-04-01',
                'deadline': '2025-04-10',
                'status': 'Assigned', # Assigned (Not Started) task
                'assigned_to_id': 'EMP-001',
                'assigned_to_name': 'Alice Johnson',
                'required_skillsets': 'Security; Python; AWS',
                'created_date': '2025-02-20 09:00:00'
            },

            # --- Bob Chen (EMP-002) - Heavy Workload ---
            {
                'task_id': 'TASK-1003',
                'project_id': 'PROJ-001',
                'task_name': 'Create Dashboard UI',
                'task_description': 'Build the main analytics dashboard with interactive charts and real-time updates.',
                'complexity': 3,
                'category': 'Frontend',
                'estimated_duration': 25,
                'estimated_budget': 7500,
                'start_date': '2025-02-10',
                'deadline': '2025-04-15',
                'status': 'In Progress',
                'assigned_to_id': 'EMP-002',
                'assigned_to_name': 'Bob Chen',
                'required_skillsets': 'React; JavaScript; CSS',
                'created_date': '2025-01-25 14:00:00'
            },
            {
                'task_id': 'TASK-2002',
                'project_id': 'PROJ-002',
                'task_name': 'Implement Product Catalog',
                'task_description': 'Develop the new product catalog component with filtering and search functionality.',
                'complexity': 4,
                'category': 'Frontend',
                'estimated_duration': 18,
                'estimated_budget': 6000,
                'start_date': '2025-03-01',
                'deadline': '2025-03-31',
                'status': 'In Progress',
                'assigned_to_id': 'EMP-002',
                'assigned_to_name': 'Bob Chen',
                'required_skillsets': 'React; JavaScript; CSS',
                'created_date': '2025-02-10 11:00:00'
            },
            {
                'task_id': 'TASK-2003',
                'project_id': 'PROJ-002',
                'task_name': 'Mobile Responsive Layout',
                'task_description': 'Ensure all pages are fully responsive on mobile devices.',
                'complexity': 3,
                'category': 'Frontend',
                'estimated_duration': 12,
                'estimated_budget': 4000,
                'start_date': '2025-02-15',
                'deadline': '2025-02-28',
                'status': 'Completed',
                'assigned_to_id': 'EMP-002',
                'assigned_to_name': 'Bob Chen',
                'required_skillsets': 'CSS; Mobile Design',
                'created_date': '2025-02-01 10:00:00'
            },

            # --- Carol Davis (EMP-003) ---
            {
                'task_id': 'TASK-1002',
                'project_id': 'PROJ-001',
                'task_name': 'Build ML Models',
                'task_description': 'Develop machine learning models for predictive analytics and anomaly detection.',
                'complexity': 5,
                'category': 'Data Science',
                'estimated_duration': 30,
                'estimated_budget': 12000,
                'start_date': '2025-02-01',
                'deadline': '2025-04-30',
                'status': 'In Progress',
                'assigned_to_id': 'EMP-003',
                'assigned_to_name': 'Carol Davis',
                'required_skillsets': 'Python; Machine Learning; TensorFlow',
                'created_date': '2025-01-20 10:30:00'
            },
            {
                'task_id': 'TASK-1006',
                'project_id': 'PROJ-001',
                'task_name': 'Model Validation Pipeline',
                'task_description': 'Create automated tests to validate ML model accuracy over time.',
                'complexity': 4,
                'category': 'Data Science',
                'estimated_duration': 14,
                'estimated_budget': 5000,
                'start_date': '2025-05-01',
                'deadline': '2025-05-15',
                'status': 'Assigned',
                'assigned_to_id': 'EMP-003',
                'assigned_to_name': 'Carol Davis',
                'required_skillsets': 'Python; Data Analysis',
                'created_date': '2025-02-25 10:30:00'
            },

            # --- David Wilson (EMP-004) - Mixed Status (No Blocked) ---
            {
                'task_id': 'TASK-3001',
                'project_id': 'PROJ-003',
                'task_name': 'Infrastructure Assessment',
                'task_description': 'Assess current infrastructure and create migration plan.',
                'complexity': 4,
                'category': 'DevOps',
                'estimated_duration': 20,
                'estimated_budget': 8000,
                'start_date': '2025-03-15',
                'deadline': '2025-04-15',
                'status': 'In Progress',
                'assigned_to_id': 'EMP-004',
                'assigned_to_name': 'David Wilson',
                'required_skillsets': 'AWS; Linux; Docker',
                'created_date': '2025-02-20 13:00:00'
            },
            {
                'task_id': 'TASK-3002',
                'project_id': 'PROJ-003',
                'task_name': 'Configure Auto-scaling',
                'task_description': 'Set up auto-scaling groups for the new cloud environment.',
                'complexity': 5,
                'category': 'DevOps',
                'estimated_duration': 10,
                'estimated_budget': 4500,
                'start_date': '2025-04-16',
                'deadline': '2025-04-26',
                'status': 'Assigned', # Changed from Blocked to Assigned
                'assigned_to_id': 'EMP-004',
                'assigned_to_name': 'David Wilson',
                'required_skillsets': 'AWS; Kubernetes',
                'created_date': '2025-02-25 13:00:00'
            },
             {
                'task_id': 'TASK-3003',
                'project_id': 'PROJ-003',
                'task_name': 'CI/CD Pipeline Setup',
                'task_description': 'Implement Jenkins pipelines for automated deployment.',
                'complexity': 4,
                'category': 'DevOps',
                'estimated_duration': 15,
                'estimated_budget': 6000,
                'start_date': '2025-05-01',
                'deadline': '2025-05-15',
                'status': 'Assigned',
                'assigned_to_id': 'EMP-004',
                'assigned_to_name': 'David Wilson',
                'required_skillsets': 'Jenkins; CI/CD',
                'created_date': '2025-03-01 13:00:00'
            },

            # --- Grace Kim (EMP-007) ---
            {
                'task_id': 'TASK-2001',
                'project_id': 'PROJ-002',
                'task_name': 'User Research & Wireframes',
                'task_description': 'Conduct user research and create wireframes for the new e-commerce design.',
                'complexity': 3,
                'category': 'Design',
                'estimated_duration': 14,
                'estimated_budget': 4500,
                'start_date': '2025-02-05',
                'deadline': '2025-02-28',
                'status': 'Completed',
                'assigned_to_id': 'EMP-007',
                'assigned_to_name': 'Grace Kim',
                'required_skillsets': 'Figma; User Research; Prototyping',
                'created_date': '2025-02-01 09:00:00'
            },
            {
                'task_id': 'TASK-2004',
                'project_id': 'PROJ-002',
                'task_name': 'High Fidelity Mockups',
                'task_description': 'Create pixel-perfect mockups for the checkout flow.',
                'complexity': 4,
                'category': 'Design',
                'estimated_duration': 10,
                'estimated_budget': 3500,
                'start_date': '2025-03-01',
                'deadline': '2025-03-10',
                'status': 'Completed',
                'assigned_to_id': 'EMP-007',
                'assigned_to_name': 'Grace Kim',
                'required_skillsets': 'Figma; UI Design',
                'created_date': '2025-02-20 09:00:00'
            },

            # --- Frank Lee (EMP-006) ---
            {
                'task_id': 'TASK-4001',
                'project_id': 'PROJ-004',
                'task_name': 'Backend API Development',
                'task_description': 'Develop REST APIs for customer portal features.',
                'complexity': 3,
                'category': 'Backend',
                'estimated_duration': 15,
                'estimated_budget': 5000,
                'start_date': '2025-01-15',
                'deadline': '2025-02-20',
                'status': 'Completed',
                'assigned_to_id': 'EMP-006',
                'assigned_to_name': 'Frank Lee',
                'required_skillsets': 'Java; Spring; SQL',
                'created_date': '2025-01-10 10:00:00'
            },
             {
                'task_id': 'TASK-4003',
                'project_id': 'PROJ-004',
                'task_name': 'API Documentation',
                'task_description': 'Document all API endpoints using Swagger.',
                'complexity': 2,
                'category': 'Backend',
                'estimated_duration': 5,
                'estimated_budget': 1500,
                'start_date': '2025-02-21',
                'deadline': '2025-02-26',
                'status': 'Completed',
                'assigned_to_id': 'EMP-006',
                'assigned_to_name': 'Frank Lee',
                'required_skillsets': 'Swagger; Documentation',
                'created_date': '2025-02-15 10:00:00'
            },

            # --- Henry Brown (EMP-008) ---
            {
                'task_id': 'TASK-4002',
                'project_id': 'PROJ-004',
                'task_name': 'Frontend Integration',
                'task_description': 'Integrate frontend with new backend APIs and optimize performance.',
                'complexity': 2,
                'category': 'Frontend',
                'estimated_duration': 10,
                'estimated_budget': 3500,
                'start_date': '2025-02-25',
                'deadline': '2025-03-15',
                'status': 'Completed',
                'assigned_to_id': 'EMP-008',
                'assigned_to_name': 'Henry Brown',
                'required_skillsets': 'React; JavaScript',
                'created_date': '2025-01-15 14:00:00'
            },
            {
                'task_id': 'TASK-4004',
                'project_id': 'PROJ-004',
                'task_name': 'Unit Testing',
                'task_description': 'Write unit tests for the frontend components.',
                'complexity': 3,
                'category': 'Testing',
                'estimated_duration': 7,
                'estimated_budget': 2000,
                'start_date': '2025-03-16',
                'deadline': '2025-03-23',
                'status': 'Assigned',
                'assigned_to_id': 'EMP-008',
                'assigned_to_name': 'Henry Brown',
                'required_skillsets': 'Jest; React Testing Library',
                'created_date': '2025-03-10 14:00:00'
            }
        ]
        
        # Sample Notifications
        sample_notifications = [
            {
                'notification_id': 'NOT-001',
                'message': 'New project created: Quantum Analytics Platform',
                'type': 'info',
                'project_id': 'PROJ-001',
                'task_id': None,
                'timestamp': '2025-01-10 09:05:00',
                'read': True
            },
            {
                'notification_id': 'NOT-002',
                'message': 'Task TASK-1001 completed by Alice Johnson',
                'type': 'info',
                'project_id': 'PROJ-001',
                'task_id': 'TASK-1001',
                'timestamp': '2025-02-05 16:30:00',
                'read': True
            },
            {
                'notification_id': 'NOT-003',
                'message': 'High priority issue reported: Performance degradation in data pipeline',
                'type': 'warning',
                'project_id': 'PROJ-001',
                'task_id': 'TASK-1001',
                'timestamp': '2025-02-10 14:25:00',
                'read': False
            }
        ]
        
        # Add all sample data to session state
        st.session_state.employees_data = sample_employees
        st.session_state.projects_data = sample_projects
        st.session_state.tasks_data = sample_tasks
        st.session_state.notifications_data = sample_notifications
        
        return True

    def render_sidebar(self):
        """Render Sidebar"""
        st.sidebar.header("ℹ️ System Overview")
        st.sidebar.write(f"**Employees:** {len(st.session_state.employees_data)}")
        st.sidebar.write(f"**Projects:** {len(st.session_state.projects_data)}")
        st.sidebar.write(f"**Tasks:** {len(st.session_state.tasks_data)}")
        st.sidebar.write(f"**Documents:** {len(st.session_state.documents_data)}")
        st.sidebar.write(f"**Open Issues:** {len([i for i in st.session_state.issues_data if i['status'] == 'Open'])}")
    
        st.sidebar.header("🚀 Quick Actions")
        if st.sidebar.button("Clear All Data"):
            st.session_state.employees_data = []
            st.session_state.projects_data = []
            st.session_state.tasks_data = []
            st.session_state.documents_data = []
            st.session_state.issues_data = []
            st.session_state.notifications_data = []
            st.rerun()
    
        if st.sidebar.button("Generate Sample Data"):
            if self.generate_sample_data():
                st.sidebar.success("Comprehensive sample data generated!")
                st.rerun()
            else:
                st.sidebar.error("Failed to generate sample data")

    # --- Main Run Method ---
    
    def run(self):
        """Main method to run the application"""
        # Set up the page
        st.set_page_config(
            page_title="Collaborative Workspace System",
            page_icon="🚀",
            layout="wide"
        )
        
        # Title and introduction
        st.title("🚀 Collaborative Workspace System")
        st.markdown("""
        Comprehensive collaborative workspace system with employee skill matching, progress tracking, and document versioning.
        """)
        
        # Use radio buttons，automatically refresh when switching
        tab_options = [
            "🏠 Dashboard",
            "👥 Employee Management", 
            "📋 Project Management",
            "✅ Task Management", 
            "📊 Analytics & Reports",
            "📁 Documents & Issues"
        ]
        
        selected_tab = st.radio(
            "Navigation",
            tab_options,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Render corresponding content based on selection
        if selected_tab == "🏠 Dashboard":
            self.render_dashboard()
        elif selected_tab == "👥 Employee Management":
            self.render_employee_management()
        elif selected_tab == "📋 Project Management":
            self.render_project_workflow()
        elif selected_tab == "✅ Task Management":
            self.render_task_workflow()
        elif selected_tab == "📊 Analytics & Reports":
            self.render_analytics_reports()
        elif selected_tab == "📁 Documents & Issues":
            self.render_documents_issues()
        
        # Render sidebar
        self.render_sidebar()

# Main execution
if __name__ == "__main__":
    pms = ProjectManagementSystem()
    pms.run()
