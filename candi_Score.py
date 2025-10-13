from integration_pipeline import IntegratedJobFitnessScorer

# Give your file paths here 👇
resume_path = "/Users/prakharbhardwaj/Desktop/Job_fitness_scoring_system/extraction/output/resume/amazon-data-science-resume-example.txt"
jd_path = "/Users/prakharbhardwaj/Desktop/Job_fitness_scoring_system/extraction/output/jd/Data_Scientist_Job_Description.txt"

# Initialize scorer
scorer = IntegratedJobFitnessScorer()

# Get result for one candidate
result = scorer.score_candidate(resume_path, jd_path)

# Print result
print("\n=== Candidate Scoring Result ===")
print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Hire Probability: {result['hire_probability']:.1%}")
print(f"Reject Probability: {result['reject_probability']:.1%}")
print(f"Similarity Score: {result['similarity_score']:.2f}")

print("\nKey Features:")
for k, v in result['key_features'].items():
    print(f"  {k}: {v}")
