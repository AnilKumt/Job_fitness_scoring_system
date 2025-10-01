"""
Integration Pipeline - Connects extraction system with ML model
This bridges your existing extraction/ code with the ML training pipeline
"""

import pandas as pd
import sys
sys.path.append('./extraction')
sys.path.append('./src')

from extract import main as extract_resume
from similarity import compute_similarity
from data_parser import DataParser
from feature_engineering import FeatureEngineer
from model import JobFitnessModel

class IntegratedJobFitnessScorer:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = None
        
    def process_resume_and_jd(self, resume_txt_path, jd_txt_path):
        """
        Process resume and job description to create features for ML model
        
        Args:
            resume_txt_path: Path to extracted resume text
            jd_txt_path: Path to extracted job description text
            
        Returns:
            dict: Processed features ready for ML model
        """
        # Extract data using your existing pipeline
        resume_data = extract_resume(resume_txt_path)
        jd_data = extract_resume(jd_txt_path)
        
        # Calculate similarity score (your existing function)
        similarity_score = compute_similarity(resume_data, jd_data)
        
        # Create feature dictionary matching your ML model expectations
        features = {
            'Skills': ', '.join(resume_data['skills']),
            'Experience (Years)': resume_data['experience'],
            'Education': resume_data['highest_degree'] if resume_data['highest_degree'] else 'None',
            'Certifications': 'None',  # Add this to your extraction if needed
            'Job Role': self._infer_job_role(jd_data['skills']),
            'Projects Count': self._estimate_projects(resume_data),
            'AI Score (0-100)': int(similarity_score * 100),
            'similarity_score': similarity_score
        }
        
        return features
    
    def _infer_job_role(self, jd_skills):
        """Infer job role from JD skills"""
        skills_str = ' '.join(jd_skills).lower()
        
        if 'data scientist' in skills_str or 'machine learning' in skills_str:
            return 'Data Scientist'
        elif 'software' in skills_str or 'developer' in skills_str:
            return 'Software Engineer'
        elif 'ai' in skills_str or 'deep learning' in skills_str:
            return 'AI Researcher'
        elif 'security' in skills_str or 'cybersecurity' in skills_str:
            return 'Cybersecurity Analyst'
        else:
            return 'Data Scientist'  # Default
    
    def _estimate_projects(self, resume_data):
        """Estimate project count from experience and skills"""
        exp_years = resume_data['experience']
        skill_count = len(resume_data['skills'])
        
        # Heuristic: ~1 project per year + bonus for many skills
        estimated_projects = int(exp_years) + (skill_count // 5)
        return min(estimated_projects, 10)  # Cap at 10
    
    def convert_to_dataframe(self, features_dict):
        """Convert feature dict to DataFrame for ML model"""
        return pd.DataFrame([features_dict])
    
    def score_candidate(self, resume_txt_path, jd_txt_path, model_path='models/job_fitness_model.pkl'):
        """
        End-to-end scoring pipeline
        
        Args:
            resume_txt_path: Path to resume text file
            jd_txt_path: Path to JD text file
            model_path: Path to trained model
            
        Returns:
            dict: Scoring results with probability and features
        """
        # Extract features
        features = self.process_resume_and_jd(resume_txt_path, jd_txt_path)
        
        # Convert to DataFrame
        df = self.convert_to_dataframe(features)
        
        # Engineer features (using your existing pipeline)
        X_engineered = self.feature_engineer.create_all_features(df)
        X_final = self.feature_engineer.get_model_features(X_engineered)
        X_scaled = self.feature_engineer.scale_features(X_final, fit=False)
        
        # Load model and predict
        model = JobFitnessModel()
        model.load_model(model_path)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        result = {
            'decision': 'HIRE' if prediction == 1 else 'REJECT',
            'confidence': float(probability[prediction]),
            'hire_probability': float(probability[1]),
            'reject_probability': float(probability[0]),
            'similarity_score': features['similarity_score'],
            'key_features': {
                'skills': features['Skills'],
                'experience': features['Experience (Years)'],
                'education': features['Education'],
                'ai_score': features['AI Score (0-100)']
            }
        }
        
        return result


def create_training_dataset_from_extractions():
    """
    Create training dataset from your existing extracted resumes
    Uses your dataset.csv as a starting point
    """
    # Load your existing dataset
    df = pd.read_csv('extraction/dataset.csv')
    
    # Add missing columns needed for ML model
    df['Certifications'] = 'None'
    df['Job Role'] = df['skills'].apply(lambda x: 'Data Scientist' if 'python' in str(x).lower() else 'Software Engineer')
    df['Projects Count'] = df['experience_years'].apply(lambda x: int(x) if x > 0 else 0)
    df['AI Score (0-100)'] = 70  # Default score, you can improve this
    df['Recruiter Decision'] = df['label'].apply(lambda x: 'Hire' if x == 1 else 'Reject')
    
    # Rename columns to match ML model expectations
    df_ml = df.rename(columns={
        'skills': 'Skills',
        'experience_years': 'Experience (Years)',
        'education_degree': 'Education'
    })
    
    # Select relevant columns
    ml_columns = ['Skills', 'Experience (Years)', 'Education', 'Certifications', 
                  'Job Role', 'Projects Count', 'AI Score (0-100)', 'Recruiter Decision']
    
    df_final = df_ml[ml_columns]
    
    # Save to data/raw folder
    df_final.to_csv('data/raw/AI_RESUME_SCREENING.csv', index=False)
    print(f"Created training dataset with {len(df_final)} samples")
    print(f"Saved to: data/raw/AI_RESUME_SCREENING.csv")
    
    return df_final


if __name__ == "__main__":
    # Step 1: Create training dataset from your extractions
    print("="*60)
    print("STEP 1: Creating Training Dataset")
    print("="*60)
    df = create_training_dataset_from_extractions()
    
    # Step 2: Test the integrated pipeline
    print("\n" + "="*60)
    print("STEP 2: Testing Integrated Pipeline")
    print("="*60)
    
    scorer = IntegratedJobFitnessScorer()
    
    # Test with your sample files
    result = scorer.score_candidate(
        'extraction/output/resume/Resume-Sample-2.txt',
        'extraction/output/jd/Data_Scientist_Job_Description.txt'
    )
    
    print("\nScoring Result:")
    print(f"Decision: {result['decision']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Hire Probability: {result['hire_probability']:.2%}")
    print(f"Similarity Score: {result['similarity_score']:.3f}")