"""
Integration Pipeline - Connects extraction system with ML model
This bridges your existing extraction/ code with the ML training pipeline
"""
import pandas as pd
import sys
import os

# --- Robust Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'extraction'))
sys.path.append(os.path.join(script_dir, 'src'))

from extract import main as extract_resume
from similarity import compute_similarity
from feature_engineering import FeatureEngineer
from model import JobFitnessModel

class IntegratedJobFitnessScorer:
    def __init__(self, model_path='models/xgboost_job_fitness.pkl', fe_path='models/feature_engineer.pkl'):
        print("Initializing scorer...")
        if not os.path.exists(model_path) or not os.path.exists(fe_path):
            raise FileNotFoundError(
                "Model or Feature Engineer not found. Please run train_complete_model.py first."
            )
        self.feature_engineer = FeatureEngineer.load(fe_path)
        self.model = JobFitnessModel(model_type='xgboost')
        self.model.load_model(model_path)
        print("Scorer initialized successfully.")

    def process_resume_and_jd(self, resume_txt_path, jd_txt_path):
        resume_data = extract_resume(resume_txt_path)
        jd_data = extract_resume(jd_txt_path)
        similarity_score = compute_similarity(resume_data, jd_data)
        features = {
            'Skills': ', '.join(resume_data.get('skills', [])),
            'Experience (Years)': resume_data.get('experience', 0),
            'Education': resume_data.get('highest_degree') or 'None',
            'Certifications': 'None',
            'Job Role': self._infer_job_role(jd_data.get('skills', [])),
            'Projects Count': self._estimate_projects(resume_data),
            'AI Score (0-100)': int(similarity_score * 100)
        }
        return features, similarity_score

    def _infer_job_role(self, jd_skills):
        skills_str = ' '.join(jd_skills if jd_skills else []).lower()
        if 'software' in skills_str or 'developer' in skills_str: return 'Software Engineer'
        if 'ai' in skills_str or 'deep learning' in skills_str: return 'AI Researcher'
        if 'security' in skills_str or 'cybersecurity' in skills_str: return 'Cybersecurity Analyst'
        return 'Data Scientist'

    def _estimate_projects(self, resume_data):
        exp_years = resume_data.get('experience', 0)
        skill_count = len(resume_data.get('skills', []))
        return min(int(exp_years) + (skill_count // 5), 10)

    def score_candidate(self, resume_txt_path, jd_txt_path):
        features, similarity_score = self.process_resume_and_jd(resume_txt_path, jd_txt_path)
        df = pd.DataFrame([features])
        X_engineered = self.feature_engineer.create_all_features(df)
        X_final = self.feature_engineer.get_model_features(X_engineered)

        model_feature_names = self.model.feature_names
        if model_feature_names:
            for col in model_feature_names:
                if col not in X_final.columns:
                    X_final[col] = 0
            X_final = X_final[model_feature_names]

        X_scaled = self.feature_engineer.scale_features(X_final, fit=False)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]

        return {
            'decision': 'HIRE' if prediction == 1 else 'REJECT',
            'confidence': float(probability[prediction]),
            'hire_probability': float(probability[1]),
            'reject_probability': float(probability[0]),
            'similarity_score': similarity_score,
            'key_features': {k: features[k] for k in ['Skills', 'Experience (Years)', 'Education', 'AI Score (0-100)']}
        }

if __name__ == "__main__":
    print("="*60 + "\nSTEP: Testing Integrated Pipeline\n" + "="*60)
    try:
        scorer = IntegratedJobFitnessScorer()
        result = scorer.score_candidate(
            'extraction/output/resume/Resume-Sample-2.txt',
            'extraction/output/jd/Data_Scientist_Job_Description.txt'
        )
        print("\nScoring Result:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
            elif isinstance(value, dict):
                print(f"  {key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    print(f"    - {k}: {v}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    except (FileNotFoundError, Exception) as e:
        print(f"\nAn error occurred: {e}")