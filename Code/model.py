import pandas as pd
from typing import List, Dict, Optional, Union
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MultiLabelBinarizer
import os

class TransformerSkillPredictor(nn.Module):
    def __init__(self, n_classes, model_name='bert-base-uncased'):
        super(TransformerSkillPredictor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class DeepSkillPredictor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlb = MultiLabelBinarizer()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
    def predict_skills(self, description, threshold=0.3):
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        
        encoding = self.tokenizer(
            description,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        predicted_skills = []
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                skill = self.mlb.classes_[i]
                predicted_skills.append((skill, prob))
        
        predicted_skills.sort(key=lambda x: x[1], reverse=True)
        
        return [skill for skill, prob in predicted_skills]
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.mlb = checkpoint['mlb']
        self.tokenizer = checkpoint['tokenizer']
        self.model = TransformerSkillPredictor(len(self.mlb.classes_))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
def get_skills_from_description(description):
    model_path = 'deep_skill_model.pth'
    if os.path.exists(model_path):
        deep_predictor = DeepSkillPredictor(model_name='distilbert-base-uncased')
        deep_predictor.load_model(model_path)
        skill = deep_predictor.predict_skills(description, threshold=0.2)
        return skill

def predict_budget(complexity_level, duration_model_path='duration_model.pkl', budget_model_path='budget_model.pkl'):
    if not (1 <= complexity_level <= 5):
        return {"error": "Complexity must be between 1 and 5"}
    
    try:
        duration_model = joblib.load(duration_model_path)
        budget_model = joblib.load(budget_model_path)
    except FileNotFoundError:
        return {"error": "Model files not found. Run training first."}
    
    X_pred = pd.DataFrame([[complexity_level]], columns=['complexity'])
    
    try:
        duration_pred = duration_model.predict(X_pred)[0]
        budget_pred = budget_model.predict(X_pred)[0]
        
        return int(duration_pred),int(budget_pred),
    except Exception as e:
        return "error",str(e)

def select_optimal_team(
    required_skills: Union[str, List[str]], 
    employees_data: List[Dict], 
    project_complexity: Optional[float] = None, 
    min_match_threshold: float = 0.6
) -> List[Dict]:

    if isinstance(required_skills, str):
        required_skills_list = [skill.strip() for skill in required_skills.split(';')] if required_skills else []
    else:
        required_skills_list = required_skills
    
    if not required_skills_list:
        return []
    
    def parse_skills(skills_str: str) -> Dict[str, int]:
        if pd.isna(skills_str) or not isinstance(skills_str, str):
            return {}
        skills_dict = {}
        skill_pairs = skills_str.split('; ')
        for pair in skill_pairs:
            if ':' in pair:
                skill, proficiency = pair.split(':')
                try:
                    skills_dict[skill.strip()] = int(proficiency)
                except (ValueError, TypeError):
                    continue
        return skills_dict
    
    employees_df = pd.DataFrame(employees_data)
    
    required_columns = ['skills', 'performance_rating', 'experience_years']
    for col in required_columns:
        if col not in employees_df.columns:
            employees_df[col] = 0 if col != 'skills' else ''
    
    employee_scores = []
    
    for _, employee in employees_df.iterrows():
        employee_data = employee.to_dict()
        skills_dict = parse_skills(employee_data.get('skills', ''))
        
        matched_skills = set(skills_dict.keys()) & set(required_skills_list)
        skill_match_score = len(matched_skills) / len(required_skills_list) if required_skills_list else 0
        
        if skill_match_score < min_match_threshold:
            continue
        
        total_proficiency = sum(skills_dict[skill] for skill in matched_skills)
        matched_count = len(matched_skills)
        avg_proficiency = total_proficiency / matched_count if matched_count > 0 else 0
        
        all_proficiencies = list(skills_dict.values())
        proficiency_cap = max(all_proficiencies) if all_proficiencies else 10
        skill_proficiency_score = total_proficiency / (matched_count * proficiency_cap) if matched_count > 0 else 0
        \
        performance_rating = employee_data.get('performance_rating', 0)
        experience_years = employee_data.get('experience_years', 0)
        
        performance_score = performance_rating / 5.0 if performance_rating else 0
        experience_score = min(experience_years / 20.0, 1.0) if experience_years else 0
        
        weights = {'skill_match': 0.4, 'skill_proficiency': 0.3, 'performance': 0.2, 'experience': 0.1}
        total_score = sum(
            weights[k] * v for k, v in {
                'skill_match': skill_match_score,
                'skill_proficiency': skill_proficiency_score,
                'performance': performance_score,
                'experience': experience_score
            }.items()
        )
        
        if project_complexity is not None and project_complexity >= 7 and experience_years:
            experience_bonus = min(experience_years / 10 * 0.1, 0.2)
            total_score += experience_bonus
        
        total_score = min(total_score, 1.0)
        
        employee_scores.append({
            'employee_data': employee_data,
            'total_score': total_score,
            'matched_skills': list(matched_skills),
            'skill_match_score': skill_match_score,
            'avg_proficiency': avg_proficiency
        })
    
    if not employee_scores:
        return []
    
    employee_scores.sort(key=lambda x: x['total_score'], reverse=True)
    
    selected_team = []
    covered_skills = set()
    remaining_candidates = employee_scores.copy()
    
    while len(selected_team) < 6 and remaining_candidates:
        best_candidate = max(
            remaining_candidates,
            key=lambda c: len(set(c['matched_skills']) - covered_skills)
        )
        
        if best_candidate:
            selected_team.append(best_candidate)
            covered_skills.update(best_candidate['matched_skills'])
            remaining_candidates.remove(best_candidate)
    
    return [
        {
            'employee': member['employee_data'],
            'match_ratio': member['skill_match_score'],
            'match_score': member['total_score'],
            'matched_skills': member['matched_skills'],
            'avg_proficiency': member['avg_proficiency']
        }
        for member in selected_team
    ]