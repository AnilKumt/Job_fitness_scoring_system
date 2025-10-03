import spacy
import sys
from utils import *



def main(file_name):
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    text = text.replace('â€“', '-')

    nlp = spacy.blank("en")
    nlp = custom_tokenizer(nlp)
    ruler = nlp.add_pipe('entity_ruler')

    with open('data/skills.txt', 'r') as f:
        skills = f.readlines()
    skills = [skill.strip() for skill in skills]

    with open('data/soft_skills.txt', 'r') as f:
        soft_skills = f.readlines()
    soft_skills = [skill.strip() for skill in soft_skills]

    with open('data/education.txt', 'r') as f:
        education = f.readlines()
    education = [edu.strip() for edu in education]

    patterns = []
    for skill in skills:
        tokens = skill.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "SKILL", "pattern": pattern})

    for s in soft_skills:
        tokens = s.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "SOFT_SKILL", "pattern": pattern})

    for edu in education:
        tokens = edu.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "EDUCATION", "pattern": pattern})

    ruler.add_patterns(patterns)

    doc = nlp(text)

    d = {
    'education': [],
    'skills': [],
    'experience': 0.0,
    'highest_degree': None,
    'soft_skills': []
    }

    skills = set()
    edu = set()
    soft = set()

    for ent in doc.ents:
        label = ent.label_.lower()

        if label == 'education':
            if ent.text.lower() not in edu:
                edu.add(ent.text.lower())
                d['education'].append(ent.text.lower())
        elif label == 'skill':
            skills.add(ent.text.lower())
        elif label == 'soft_skill':
            soft.add(ent.text.lower())
        
    d['skills'] = sorted(list(skills))
    d['experience'] = extract_experience(text)
    # Liberty taken than user keeps his degrees from highest to lowest.
    d['highest_degree'] = d['education'][0] if d['education'] else None
    d['soft_skills'] = sorted(list(soft))

    d['output_file'] = file_name

    return d

    # print(d)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python extract.py <txt_file> <csv_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    output_csv = sys.argv[2]
    d = main(file_name)
    write_to_csv(d, output_csv)