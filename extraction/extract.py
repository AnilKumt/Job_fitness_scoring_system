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

    with open('data/education.txt', 'r') as f:
        education = f.readlines()
    education = [edu.strip() for edu in education]

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

    doc = nlp(text)

    d = {
    'education': [],
    'skills': [],
    'experience': 0.0,
    'highest_degree': None
    }

    skills = set()
    edu = set()

    for ent in doc.ents:
        label = ent.label_.lower()

        if label == 'education':
            if ent.text.lower() not in edu:
                edu.add(ent.text.lower())
                d['education'].append(ent.text.lower())
        elif label == 'skill':
            skills.add(ent.text.lower())
        
    d['skills'] = list(skills)
    d['experience'] = extract_experience(text)
    # Liberty taken than user keeps his degrees from highest to lowest.
    d['highest_degree'] = d['education'][0] if d['education'] else None

    write_to_csv(d, 'dataset.csv')

    # print(d)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python extract.py <txt_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    main(file_name)