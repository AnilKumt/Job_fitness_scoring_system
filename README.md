# Job_fitness_scoring_system
# Job Fitness Scoring System

An ML-Powered automated candidate screening system.

## Team
- Prakhar Bharadhwaj (123cs0071)
- Manish (123cs0005)
- Anil Kumawat (123cs0051)

## Setup
```bash
python3 -m venv ml_env
source ml_env/bin/activate
pip install -r requirements.txt

python -m spacy download en_core_web_sm
pip install sentence-transformers

COMMANDS->

run data_augmentation.py
run train_complete_model.py
run integration_pipeline.py
opt1> add the resume path to Candi_Score.py for single resume ai score .
      run Candi_score.py
opt2> add the resume dic to the score_candidate.py to find the best options of the reumes/ candidate
      run score_candidate.py

