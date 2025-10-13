"""
Integration Pipeline - FINAL VERSION
Connects extraction system with ML model
"""

import pandas as pd
import sys
import os
import joblib

sys.path.append('./extraction')
sys.path.append('./src')

from extract import main as extract_resume
from similarity import compute_similarity
from data_parser import DataParser
from feature_engineering import FeatureEngineer
from model import JobFitnessModel


class IntegratedJobFitnessScorer:
    def __init__(self, model_path='models/xgboost_job_fitness.pkl', 
                 engineer_path='models/feature_engineer.pkl'):
        """
        Initialize the integrated scorer
        
        Args:
            model_path: Path to trained model
            engineer_path: Path to trained feature engineer
        """
        # Load trained feature engineer (with fitted scaler)
        if os.path.exists(engineer_path):
            self.feature_engineer = joblib.load(engineer_path)
            print(f"Loaded feature engineer from {engineer_path}")
            if hasattr(self.feature_engineer.scaler, 'n_features_in_'):
                print(f"Scaler was fitted on {self.feature_engineer.scaler.n_features_in_} features")
        else:
            print(f"Warning: Feature engineer not found at {engineer_path}")
            self.feature_engineer = FeatureEngineer()
        
        # Load trained model
        self.model = JobFitnessModel()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            self.expected_features = self.model.feature_names
            print(f"Model expects {len(self.expected_features)} features")
        else:
            print(f"Warning: Model not found at {model_path}")
            self.model = None
            self.expected_features = None
        
    def process_resume_and_jd(self, resume_txt_path, jd_txt_path):
        """Process resume and job description to create features"""
        if not os.path.exists(resume_txt_path):
            raise FileNotFoundError(f"Resume file not found: {resume_txt_path}")
        if not os.path.exists(jd_txt_path):
            raise FileNotFoundError(f"JD file not found: {jd_txt_path}")
        
        # Extract data
        resume_data = extract_resume(resume_txt_path)
        jd_data = extract_resume(jd_txt_path)
        
        # Calculate similarity
        similarity_score = compute_similarity(resume_data, jd_data)
        
        # Create features
        features = {
            'Skills': ', '.join(resume_data['skills']) if resume_data['skills'] else 'None',
            'Experience (Years)': float(resume_data['experience']) if resume_data['experience'] else 0.0,
            'Education': resume_data['highest_degree'] if resume_data['highest_degree'] else 'B.Sc',
            'Certifications': 'None',
            'Job Role': self._infer_job_role(jd_data['skills']),
            'Projects Count': self._estimate_projects(resume_data),
            'AI Score (0-100)': int(similarity_score * 100),
            'similarity_score': similarity_score
        }
        
        return features
    
    def _infer_job_role(self, jd_skills):
        """Infer job role from JD skills"""
        if not jd_skills:
            return 'Data Scientist'
        
        skills_str = ' '.join(jd_skills).lower()
        
        if 'machine learning' in skills_str or 'ml' in skills_str:
            return 'Data Scientist'
        elif 'software' in skills_str or 'java' in skills_str or 'react' in skills_str:
            return 'Software Engineer'
        elif 'deep learning' in skills_str or 'pytorch' in skills_str:
            return 'AI Researcher'
        elif 'security' in skills_str or 'cybersecurity' in skills_str:
            return 'Cybersecurity Analyst'
        else:
            return 'Data Scientist'
    
    def _estimate_projects(self, resume_data):
        """Estimate project count from experience and skills"""
        exp_years = resume_data['experience'] if resume_data['experience'] else 0
        skill_count = len(resume_data['skills']) if resume_data['skills'] else 0
        estimated_projects = int(exp_years) + (skill_count // 5)
        return min(max(estimated_projects, 0), 10)
    
    def convert_to_dataframe(self, features_dict):
        """Convert feature dict to DataFrame"""
        model_features = {k: v for k, v in features_dict.items() if k != 'similarity_score'}
        return pd.DataFrame([model_features])
    
    def align_features(self, X_final):
        """Align features to match training feature names"""
        if self.expected_features is None:
            return X_final
        
        X_aligned = pd.DataFrame(
            0, 
            index=X_final.index, 
            columns=self.expected_features,
            dtype=float
        )
        
        for col in X_final.columns:
            if col in X_aligned.columns:
                X_aligned[col] = X_final[col].values
        
        return X_aligned
    
    def score_candidate(self, resume_txt_path, jd_txt_path):
        """
        End-to-end scoring pipeline
        
        Returns:
            dict: Scoring results with probability and features
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        try:
            # Extract features
            print("Extracting features from resume and JD...")
            features = self.process_resume_and_jd(resume_txt_path, jd_txt_path)
            
            # Convert to DataFrame
            df = self.convert_to_dataframe(features)
            
            # Engineer features
            print("Engineering features...")
            X_engineered = self.feature_engineer.create_all_features(df)
            X_final = self.feature_engineer.get_model_features(X_engineered)
            
            # Align features to match training
            print("Aligning features with training data...")
            X_aligned = self.align_features(X_final)
            print(f"Aligned features: {X_aligned.shape}")
            
            # Scale features
            X_scaled = self.feature_engineer.scale_features(X_aligned.copy(), fit=False)
            
            # Predict
            print("Making prediction...")
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
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
                    'job_role': features['Job Role'],
                    'ai_score': features['AI Score (0-100)']
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Error during scoring: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_training_dataset_from_extractions():
    """Create training dataset from extracted resumes"""
    try:
        df = pd.read_csv('extraction/dataset.csv')
        print(f"Loaded {len(df)} resumes from extraction/dataset.csv")
        
        # Add missing columns
        df['Certifications'] = 'None'
        
        def infer_role(skills_str):
            if pd.isna(skills_str) or skills_str == '':
                return 'Data Scientist'
            skills = str(skills_str).lower()
            if 'machine learning' in skills or 'tensorflow' in skills:
                return 'Data Scientist'
            elif 'java' in skills or 'react' in skills:
                return 'Software Engineer'
            elif 'cybersecurity' in skills:
                return 'Cybersecurity Analyst'
            else:
                return 'Data Scientist'
        
        df['Job Role'] = df['skills'].apply(infer_role)
        df['Projects Count'] = df['experience_years'].apply(lambda x: int(x) if pd.notna(x) and x > 0 else 0)
        df['AI Score (0-100)'] = 70
        df['Recruiter Decision'] = df['label'].apply(lambda x: 'Hire' if x == 1 else 'Reject')
        
        # Rename columns
        df_ml = df.rename(columns={
            'skills': 'Skills',
            'experience_years': 'Experience (Years)',
            'education_degree': 'Education'
        })
        
        # Fill NaN values
        df_ml['Skills'] = df_ml['Skills'].fillna('')
        df_ml['Experience (Years)'] = df_ml['Experience (Years)'].fillna(0)
        df_ml['Education'] = df_ml['Education'].fillna('B.Sc')
        
        # Select columns
        ml_columns = ['Skills', 'Experience (Years)', 'Education', 'Certifications', 
                      'Job Role', 'Projects Count', 'AI Score (0-100)', 'Recruiter Decision']
        
        df_final = df_ml[ml_columns]
        
        # Save
        os.makedirs('data/raw', exist_ok=True)
        df_final.to_csv('data/raw/AI_RESUME_SCREENING.csv', index=False)
        print(f"\nCreated training dataset with {len(df_final)} samples")
        print(f"Saved to: data/raw/AI_RESUME_SCREENING.csv")
        
        return df_final
        
    except FileNotFoundError:
        print("ERROR: extraction/dataset.csv not found!")
        return None


if __name__ == "__main__":
    print("="*60)
    print("INTEGRATED JOB FITNESS SCORING PIPELINE")
    print("="*60)
    
    # Step 1: Create training dataset
    print("\n[STEP 1] Creating Training Dataset from Extractions")
    print("-"*60)
    df = create_training_dataset_from_extractions()
    
    if df is None:
        print("\nFailed to create training dataset. Exiting.")
        sys.exit(1)
    
    # Step 2: Test the integrated pipeline
    print("\n[STEP 2] Testing Integrated Pipeline")
    print("-"*60)
    
    resume_path = 'extraction/output/resume/sample_resume.txt'
    jd_path = 'extraction/output/jd/Data_Scientist_Job_Description.txt'
    
    if not os.path.exists(resume_path):
        print(f"Warning: Sample resume not found at {resume_path}")
        print("Skipping pipeline test.")
    elif not os.path.exists(jd_path):
        print(f"Warning: Sample JD not found at {jd_path}")
        print("Skipping pipeline test.")
    else:
        try:
            scorer = IntegratedJobFitnessScorer()
            
            if scorer.model is not None:
                result = scorer.score_candidate(resume_path, jd_path)
                
                print("\n" + "="*60)
                print("✓ SCORING SUCCESSFUL!")
                print("="*60)
                print(f"Decision: {result['decision']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"Hire Probability: {result['hire_probability']:.2%}")
                print(f"Similarity Score: {result['similarity_score']:.3f}")
                print(f"\nKey Features:")
                for key, value in result['key_features'].items():
                    print(f"  {key}: {value}")
            else:
                print("\nModel not trained yet. Please run:")
                print("  1. python data_augmentation.py")
                print("  2. python train_complete_model.py")
                
        except Exception as e:
            print(f"\nError during pipeline test: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("PIPELINE SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python data_augmentation.py")
    print("  2. Run: python train_complete_model.py")
    print("  3. Use IntegratedJobFitnessScorer for predictions")

    # data/soft_skills.txt
