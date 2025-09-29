import spacy
import sys


def main(file_name):
    with open(file_name, 'r') as f:
        text = f.read()

    nlp = spacy.blank("en")
    ruler = nlp.add_pipe('entity_ruler')
    with open('data/names.txt', 'r') as f:
        names = f.readlines()
    names = [name.strip() for name in names]

    with open('data/skills.txt', 'r') as f:
        skills = f.readlines()
    skills = [skill.strip() for skill in skills]

    with open('data/certifications.txt', 'r') as f:
        certifications = f.readlines()
    certifications = [certi.strip() for certi in certifications]

    with open('data/education.txt', 'r') as f:
        education = f.readlines()
    education = [edu.strip() for edu in education]

    with open('data/job_roles.txt', 'r') as f:
        job_roles = f.readlines()
    job_roles = [jr.strip() for jr in job_roles]

    patterns = []
    for skill in skills:
        tokens = skill.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "SKILL", "pattern": pattern})

    for name in names:    
        tokens = name.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "NAME", "pattern": pattern})

    for edu in education:
        tokens = edu.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "EDUCATION", "pattern": pattern})

    for cer in certifications:
        pattern = [{"LOWER": cer.lower()}]
        patterns.append({"label": "CERTIFICATIONS", "pattern": pattern})

    for jr in job_roles:
        tokens = jr.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "JOB ROLE", "pattern": pattern})

    ruler.add_patterns(patterns)

    doc = nlp(text)

    d = {
    'name': '',
    'education': [],
    'certifications': [],
    'skills': [],
    'job_role': [] 
    }

    skills = set()
    jr = set()
    edu = set()

    for ent in doc.ents:
        label = ent.label_.lower()
        if label == 'name':
            d['name'] = ent.text.lower()

        elif label == 'education':
            edu.add(ent.text.lower())
        elif label == 'job role':
            jr.add(ent.text.lower())
        elif label == 'certifications':
            d['certifications'].append(ent.text.lower())
        elif label == 'skill':
            skills.add(ent.text.lower())
        
    d['skills'] = skills
    d['job_role'] = jr
    d['education'] = edu

    print(d)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python extract.py <txt_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    main(file_name)