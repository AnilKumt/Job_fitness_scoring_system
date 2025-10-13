"""
Data Augmentation Script - FIXED VERSION
Automatically assigns labels based on candidate quality
"""

import pandas as pd
import random
import numpy as np
import os

# Define skill pools
SKILL_CATEGORIES = {
    'data_science': ['python', 'sql', 'machine learning', 'tensorflow', 'pytorch', 
                     'deep learning', 'nlp', 'pandas', 'numpy', 'scikit-learn'],
    'software_eng': ['java', 'javascript', 'react', 'nodejs', 'docker', 
                     'kubernetes', 'git', 'aws', 'microservices', 'rest api'],
    'cybersecurity': ['penetration testing', 'network security', 'encryption', 
                      'siem', 'vulnerability assessment', 'firewall', 'ids/ips'],
    'ai_research': ['deep learning', 'reinforcement learning', 'computer vision',
                    'nlp', 'transformers', 'gans', 'research', 'publications']
}

EDUCATION_LEVELS = ['B.Sc', 'B.Tech', 'M.Tech', 'MBA', 'PhD']
CERTIFICATIONS = [
    'AWS Certified Solutions Architect', 'Google Cloud Professional',
    'Deep Learning Specialization', 'TensorFlow Developer Certificate',
    'Certified Ethical Hacker', 'CISSP', 'None'
]
JOB_ROLES = ['Data Scientist', 'Software Engineer', 'AI Researcher', 'Cybersecurity Analyst']


def calculate_candidate_score(row):
    """
    Calculate a score for a candidate based on their qualifications
    Higher score = more likely to hire
    """
    score = 0
    
    # Experience scoring (0-30 points)
    experience = float(row.get('experience_years', 0))
    if experience >= 5:
        score += 30
    elif experience >= 3:
        score += 20
    elif experience >= 1:
        score += 10
    else:
        score += 0
    
    # Skills scoring (0-40 points)
    skills = str(row.get('skills', '')).lower()
    if skills and skills != 'nan':
        skill_list = [s.strip() for s in skills.split(',')]
        num_skills = len(skill_list)
        
        # Number of skills
        if num_skills >= 8:
            score += 20
        elif num_skills >= 5:
            score += 15
        elif num_skills >= 3:
            score += 10
        else:
            score += 5
        
        # High-value skills bonus
        high_value_skills = ['python', 'machine learning', 'deep learning', 
                            'tensorflow', 'pytorch', 'aws', 'docker', 'kubernetes']
        for skill in high_value_skills:
            if skill in skills:
                score += 2
    
    # Education scoring (0-20 points)
    education = str(row.get('education_degree', '')).lower()
    if 'phd' in education or 'ph.d' in education:
        score += 20
    elif 'master' in education or 'm.tech' in education or 'm.s' in education or 'mba' in education:
        score += 15
    elif 'bachelor' in education or 'b.s' in education or 'b.tech' in education or 'b.sc' in education or 'b.e' in education:
        score += 10
    else:
        score += 5
    
    # Soft skills bonus (0-10 points)
    soft_skills = str(row.get('soft_skills', '')).lower()
    if soft_skills and soft_skills != 'nan':
        soft_skill_count = len(soft_skills.split(','))
        score += min(soft_skill_count, 10)
    
    return score


def assign_label_based_on_score(score, randomness=0.2):
    """
    Assign hire/reject label based on score
    
    Args:
        score: Candidate score (0-100)
        randomness: Add some randomness to make it realistic (0-1)
    
    Returns:
        1 (Hire) or 0 (Reject)
    """
    # Add random noise
    adjusted_score = score + random.uniform(-20, 20) * randomness
    
    # Decision thresholds
    if adjusted_score >= 60:
        return 1  # HIRE
    elif adjusted_score >= 45:
        # Borderline - 50% chance
        return random.choice([0, 1])
    else:
        return 0  # REJECT


def fix_labels_in_dataset(df):
    """
    Automatically assign realistic labels based on candidate quality
    """
    print("\n⚠️  WARNING: All labels are 0 (Reject)!")
    print("Automatically assigning labels based on candidate qualifications...\n")
    
    # Calculate scores for each candidate
    df['_score'] = df.apply(calculate_candidate_score, axis=1)
    
    # Assign labels based on scores
    df['label'] = df['_score'].apply(lambda s: assign_label_based_on_score(s, randomness=0.2))
    
    # Remove temporary score column
    df = df.drop('_score', axis=1)
    
    hire_count = df['label'].sum()
    reject_count = len(df) - hire_count
    
    print(f"✓ Labels assigned:")
    print(f"  - {hire_count} candidates labeled as HIRE ({hire_count/len(df)*100:.1f}%)")
    print(f"  - {reject_count} candidates labeled as REJECT ({reject_count/len(df)*100:.1f}%)")
    print()
    
    return df


def generate_augmented_candidate(base_candidate, variation_level='medium'):
    """Generate a new candidate by slightly modifying an existing one"""
    new_candidate = base_candidate.copy()
    
    # Vary experience
    exp_variation = {'low': 0.5, 'medium': 1.0, 'high': 2.0}[variation_level]
    experience = base_candidate.get('experience_years', 0)
    new_candidate['experience_years'] = max(0, experience + random.uniform(-exp_variation, exp_variation))
    
    # Vary skills
    skills_str = base_candidate.get('skills', '')
    if pd.notna(skills_str) and skills_str != '':
        skills = str(skills_str).split(', ')
        
        # Remove a random skill sometimes
        if len(skills) > 2 and random.random() > 0.5:
            skills.pop(random.randint(0, len(skills)-1))
        
        # Add random skill sometimes
        role = base_candidate.get('job_role', 'Data Scientist')
        if 'Data Scientist' in role:
            pool = SKILL_CATEGORIES['data_science']
        elif 'Software' in role:
            pool = SKILL_CATEGORIES['software_eng']
        elif 'AI' in role:
            pool = SKILL_CATEGORIES['ai_research']
        else:
            pool = SKILL_CATEGORIES['cybersecurity']
        
        if random.random() > 0.5:
            new_skill = random.choice(pool)
            if new_skill not in skills:
                skills.append(new_skill)
        
        new_candidate['skills'] = ', '.join(skills)
    
    # Vary education sometimes
    if random.random() > 0.7:
        new_candidate['education_degree'] = random.choice(EDUCATION_LEVELS)
    
    # Recalculate label based on new attributes
    score = calculate_candidate_score(new_candidate)
    new_candidate['label'] = assign_label_based_on_score(score)
    
    return new_candidate


def create_synthetic_candidate(job_role, target_label):
    """Create a completely synthetic candidate"""
    if 'Data Scientist' in job_role:
        skill_pool = SKILL_CATEGORIES['data_science']
    elif 'Software' in job_role:
        skill_pool = SKILL_CATEGORIES['software_eng']
    elif 'AI' in job_role:
        skill_pool = SKILL_CATEGORIES['ai_research']
    else:
        skill_pool = SKILL_CATEGORIES['cybersecurity']
    
    if target_label == 1:  # Strong candidate
        num_skills = random.randint(5, 8)
        experience = random.uniform(3, 10)
        projects = random.randint(3, 10)
        education = random.choice(['M.Tech', 'PhD', 'MBA'])
        has_cert = random.choice([True, True, False])
    else:  # Weak candidate
        num_skills = random.randint(1, 4)
        experience = random.uniform(0, 2)
        projects = random.randint(0, 2)
        education = random.choice(['B.Sc', 'B.Tech'])
        has_cert = random.choice([True, False, False])
    
    skills = ', '.join(random.sample(skill_pool, min(num_skills, len(skill_pool))))
    certification = random.choice(CERTIFICATIONS) if has_cert else 'None'
    
    return {
        'skills': skills,
        'experience_years': round(experience, 2),
        'education_degree': education,
        'certifications': certification,
        'job_role': job_role,
        'projects_count': projects,
        'label': target_label
    }


def augment_dataset(df, target_size=300):
    """Augment dataset to reach target size"""
    
    # CRITICAL FIX: Check and fix labels
    if 'label' not in df.columns or df['label'].sum() == 0:
        df = fix_labels_in_dataset(df)
    
    augmented_data = []
    
    # Add original data
    for _, row in df.iterrows():
        augmented_data.append({
            'skills': row.get('skills', ''),
            'experience_years': row.get('experience_years', 0),
            'education_degree': row.get('education_degree', 'B.Sc'),
            'certifications': 'None',
            'job_role': row.get('job_role', 'Data Scientist'),
            'projects_count': int(row.get('experience_years', 0)),
            'label': row.get('label', 0)
        })
    
    print(f"Original dataset: {len(augmented_data)} samples")
    
    # Augment existing candidates
    while len(augmented_data) < target_size * 0.6:
        base = random.choice(augmented_data)
        augmented = generate_augmented_candidate(base, variation_level='medium')
        augmented_data.append(augmented)
    
    print(f"After augmentation: {len(augmented_data)} samples")
    
    # Generate synthetic candidates with balanced labels
    hire_count = sum(1 for d in augmented_data if d['label'] == 1)
    target_hire_rate = 0.40  # Target 40% hire rate
    
    while len(augmented_data) < target_size:
        current_hire_rate = hire_count / len(augmented_data)
        
        if current_hire_rate < target_hire_rate:
            label = 1
            hire_count += 1
        else:
            label = 0
        
        job_role = random.choice(JOB_ROLES)
        synthetic = create_synthetic_candidate(job_role, label)
        augmented_data.append(synthetic)
    
    print(f"Final dataset: {len(augmented_data)} samples")
    print(f"Hire rate: {hire_count/len(augmented_data):.2%}")
    
    # Convert to DataFrame
    df_augmented = pd.DataFrame(augmented_data)
    
    # Compute AI Score
    df_augmented['AI Score (0-100)'] = df_augmented.apply(
        lambda row: int(20 + float(row.get('experience_years',0))*3 + len(str(row.get('skills','')).split(','))*5),
        axis=1
    )
    df_augmented['AI Score (0-100)'] = df_augmented['AI Score (0-100)'].clip(15, 100)
    
    # Recruiter Decision
    df_augmented['Recruiter Decision'] = df_augmented['label'].apply(
        lambda x: 'Hire' if x == 1 else 'Reject'
    )
    
    # Rename columns for ML model
    df_augmented = df_augmented.rename(columns={
        'skills': 'Skills',
        'experience_years': 'Experience (Years)',
        'education_degree': 'Education',
        'certifications': 'Certifications',
        'job_role': 'Job Role',
        'projects_count': 'Projects Count'
    })
    
    return df_augmented


if __name__ == "__main__":
    # Load original data
    print("="*70)
    print("DATA AUGMENTATION WITH AUTO-LABELING")
    print("="*70)
    
    df_original = pd.read_csv('extraction/dataset.csv')
    
    # Augment to 300 samples
    df_augmented = augment_dataset(df_original, target_size=300)
    
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Save
    df_augmented.to_csv('data/raw/AI_RESUME_SCREENING_AUGMENTED.csv', index=False)
    
    # Print stats
    print("\n" + "="*70)
    print("Dataset Statistics:")
    print("="*70)
    print(df_augmented[['Skills', 'Experience (Years)', 'Education', 'Recruiter Decision']].head(10))
    print(f"\nTotal samples: {len(df_augmented)}")
    print(f"Hire count: {(df_augmented['Recruiter Decision'] == 'Hire').sum()}")
    print(f"Reject count: {(df_augmented['Recruiter Decision'] == 'Reject').sum()}")
    
    # Print some examples of each class
    print("\n" + "="*70)
    print("SAMPLE HIRED CANDIDATES:")
    print("="*70)
    hired = df_augmented[df_augmented['Recruiter Decision'] == 'Hire'].head(3)
    for idx, row in hired.iterrows():
        print(f"\n{row['Skills'][:60]}...")
        print(f"Experience: {row['Experience (Years)']} years | Education: {row['Education']} | AI Score: {row['AI Score (0-100)']}")
    
    print("\n" + "="*70)
    print("SAMPLE REJECTED CANDIDATES:")
    print("="*70)
    rejected = df_augmented[df_augmented['Recruiter Decision'] == 'Reject'].head(3)
    for idx, row in rejected.iterrows():
        print(f"\n{row['Skills'][:60]}...")
        print(f"Experience: {row['Experience (Years)']} years | Education: {row['Education']} | AI Score: {row['AI Score (0-100)']}")
    
    print(f"\n{'='*70}")
    print(f"✓ Saved to: data/raw/AI_RESUME_SCREENING_AUGMENTED.csv")
    print(f"{'='*70}")
    print("\nNext step: Run 'python train_complete_model.py' to train the models")
