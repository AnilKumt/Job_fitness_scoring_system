import spacy
import sys
import os
from utils import *

def main(file_name):
    # Read resume text safely
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    text = text.replace('â€“', '-')

    # Initialize blank SpaCy model
    nlp = spacy.blank("en")
    nlp = custom_tokenizer(nlp)
    ruler = nlp.add_pipe('entity_ruler')

    # -------------------------------
    # Paths for skills and education files (robust)
    # -------------------------------
    script_dir = os.path.dirname(__file__)  # folder where extract.py is located
    skills_path = os.path.join(script_dir, 'data', 'skills.txt')
    education_path = os.path.join(script_dir, 'data', 'education.txt')

    # Ensure files exist
    if not os.path.exists(skills_path):
        raise FileNotFoundError(f"Skills file not found: {skills_path}")
    if not os.path.exists(education_path):
        raise FileNotFoundError(f"Education file not found: {education_path}")

    # Read skills
    with open(skills_path, 'r', encoding='utf-8', errors='ignore') as f:
        skills = [skill.strip() for skill in f.readlines() if skill.strip()]

    # Read education
    with open(education_path, 'r', encoding='utf-8', errors='ignore') as f:
        education = [edu.strip() for edu in f.readlines() if edu.strip()]

    # Create patterns for entity ruler
    patterns = []

    for skill in skills:
        tokens = skill.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "SKILL", "pattern": pattern})

    for edu in education:
        tokens = edu.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "EDUCATION", "pattern": pattern})

    ruler.add_patterns(patterns)

    # Process text
    doc = nlp(text)

    # Initialize result dictionary
    d = {
        'education': [],
        'skills': [],
        'experience': 0.0,
        'highest_degree': None
    }

    skills_set = set()
    edu_set = set()

    for ent in doc.ents:
        label = ent.label_.lower()
        if label == 'education' and ent.text.lower() not in edu_set:
            edu_set.add(ent.text.lower())
            d['education'].append(ent.text.lower())
        elif label == 'skill':
            skills_set.add(ent.text.lower())

    d['skills'] = sorted(list(skills_set))
    d['experience'] = extract_experience(text)  # from utils.py
    d['highest_degree'] = d['education'][0] if d['education'] else None
    d['output_file'] = file_name

    return d

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract.py <txt_file> <csv_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    output_csv = sys.argv[2]
    d = main(file_name)
    write_to_csv(d, output_csv)
