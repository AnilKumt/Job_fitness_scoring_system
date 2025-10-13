"""
Batch Candidate Scoring Script
Scores all resumes in extraction/output/resume/ against a job description
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append('.')
from integration_pipeline import IntegratedJobFitnessScorer


def score_all_candidates(resume_dir='extraction/output/resume/amazon-data-science-resume-example', 
                         jd_path='extraction/output/jd/Data_Scientist_Job_Description.txt',
                         output_file='screening_results.csv'):
    """
    Score all resume files against a job description
    
    Args:
        resume_dir: Directory containing resume .txt files
        jd_path: Path to job description .txt file
        output_file: Output CSV filename
    """
    
    print("="*70)
    print("BATCH CANDIDATE SCORING")
    print("="*70)
    
    # Check if JD exists
    if not os.path.exists(jd_path):
        print(f"\n❌ ERROR: Job description not found at {jd_path}")
        print("\nPlease extract JD first:")
        print("  cd extraction")
        print("  python batch_run_jd.py")
        return
    
    # Check if resume directory exists
    if not os.path.exists(resume_dir):
        print(f"\n❌ ERROR: Resume directory not found at {resume_dir}")
        print("\nPlease extract resumes first:")
        print("  cd extraction")
        print("  python batch_run_resume.py")
        return
    
    # Initialize scorer
    print("\n[1/3] Loading trained model...")
    try:
        scorer = IntegratedJobFitnessScorer()
    except Exception as e:
        print(f"\n❌ ERROR: Could not load model: {e}")
        print("\nPlease train the model first:")
        print("  python data_augmentation.py")
        print("  python train_complete_model.py")
        return
    
    # Get all resume files
    print(f"\n[2/3] Finding resume files in {resume_dir}...")
    resume_files = [f for f in os.listdir(resume_dir) if f.endswith('.txt')]
    
    if not resume_files:
        print(f"\n❌ ERROR: No .txt files found in {resume_dir}")
        return
    
    print(f"Found {len(resume_files)} resume files")
    
    # Score each candidate
    print(f"\n[3/3] Scoring candidates against {os.path.basename(jd_path)}...")
    results = []
    errors = []
    
    for i, resume_file in enumerate(resume_files, 1):
        resume_path = os.path.join(resume_dir, resume_file)
        
        try:
            print(f"  [{i}/{len(resume_files)}] Processing {resume_file}...", end=' ')
            
            result = scorer.score_candidate(resume_path, jd_path)
            
            results.append({
                'Candidate_File': resume_file,
                'Candidate_Name': resume_file.replace('.txt', '').replace('_', ' '),
                'Decision': result['decision'],
                'Confidence': result['confidence'],
                'Hire_Probability': result['hire_probability'],
                'Reject_Probability': result['reject_probability'],
                'Similarity_Score': result['similarity_score'],
                'Experience_Years': result['key_features']['experience'],
                'Education': result['key_features']['education'],
                'Job_Role': result['key_features']['job_role'],
                'AI_Score': result['key_features']['ai_score'],
                'Skills': result['key_features']['skills'][:100]  # Truncate long skills
            })
            
            print(f"✓ {result['decision']} ({result['confidence']:.1%})")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            errors.append({
                'file': resume_file,
                'error': str(e)
            })
    
    # Create results DataFrame
    if not results:
        print("\n❌ No candidates were successfully scored!")
        return
    
    df_results = pd.DataFrame(results)
    
    # Sort by hire probability (highest first)
    df_results = df_results.sort_values('Hire_Probability', ascending=False)
    
    # Add ranking
    df_results.insert(0, 'Rank', range(1, len(df_results) + 1))
    
    # Save to CSV
    df_results.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("SCORING COMPLETE!")
    print("="*70)
    
    total = len(results)
    hired = len(df_results[df_results['Decision'] == 'HIRE'])
    rejected = len(df_results[df_results['Decision'] == 'REJECT'])
    
    print(f"\n📊 Summary:")
    print(f"  Total Candidates: {total}")
    print(f"  Recommended HIRE: {hired} ({hired/total*100:.1f}%)")
    print(f"  Recommended REJECT: {rejected} ({rejected/total*100:.1f}%)")
    
    if errors:
        print(f"\n⚠️  Errors: {len(errors)} candidates failed to process")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print top 10 candidates
    print("\n" + "="*70)
    print("TOP 10 CANDIDATES")
    print("="*70)
    
    top_10 = df_results.head(10)[['Rank', 'Candidate_Name', 'Decision', 'Confidence', 'Experience_Years', 'AI_Score']]
    top_10['Confidence'] = top_10['Confidence'].apply(lambda x: f"{x:.1%}")
    
    print(top_10.to_string(index=False))
    
    # Print candidates to interview (high confidence HIRE)
    print("\n" + "="*70)
    print("RECOMMENDED FOR INTERVIEW (Confidence > 80%)")
    print("="*70)
    
    interview_candidates = df_results[
        (df_results['Decision'] == 'HIRE') & 
        (df_results['Confidence'] > 0.80)
    ]
    
    if len(interview_candidates) > 0:
        print(f"\n{len(interview_candidates)} candidates recommended for interview:\n")
        for _, row in interview_candidates.iterrows():
            print(f"  {row['Rank']}. {row['Candidate_Name']}")
            print(f"     Confidence: {row['Confidence']:.1%} | Experience: {row['Experience_Years']} years | AI Score: {row['AI_Score']}")
            print(f"     Skills: {row['Skills'][:80]}...")
            print()
    else:
        print("\n⚠️  No candidates meet the high confidence threshold (>80%)")
        print("Consider:")
        print("  - Lowering the confidence threshold")
        print("  - Reviewing the job description requirements")
        print("  - Adding more diverse training data")
    
    # Save error log if any
    if errors:
        error_file = output_file.replace('.csv', '_errors.txt')
        with open(error_file, 'w') as f:
            f.write("CANDIDATES WITH PROCESSING ERRORS\n")
            f.write("="*70 + "\n\n")
            for err in errors:
                f.write(f"File: {err['file']}\n")
                f.write(f"Error: {err['error']}\n\n")
        print(f"\n⚠️  Error log saved to: {error_file}")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Review screening_results.csv")
    print("  2. Schedule interviews with top candidates")
    print("  3. Provide feedback to improve the model")
    print("="*70)


def main():
    """Main function with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch score candidates against a job description')
    parser.add_argument('--resume-dir', default='extraction/output/resume/',
                        help='Directory containing resume .txt files')
    parser.add_argument('--jd', default='extraction/output/jd/Data_Scientist_Job_Description.txt',
                        help='Path to job description .txt file')
    parser.add_argument('--output', default=f'screening_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    score_all_candidates(
        resume_dir=args.resume_dir,
        jd_path=args.jd,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
