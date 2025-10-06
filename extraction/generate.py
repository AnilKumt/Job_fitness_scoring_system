# This generates required dataset from the skills, education and others

import random
import sys
from utils import *



def main(skills, soft_skills, education, file_name):
    d = {
    'education': [],
    'skills': [],
    'experience': 0.0,
    'highest_degree': None,
    'soft_skills': []
    }
    d['skills'] = random.sample(skills, k=random.randint(1, min(10, len(skills))))
    d['soft_skills'] = random.sample(soft_skills, k=random.randint(1, min(5, len(soft_skills))))
    d['highest_degree'] = random.choice(education)
    d['education'] = [d['highest_degree']]
    d['experience'] = round(random.uniform(2, 10), 2)
    

    d['output_file'] = file_name

    return d

    # print(d)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python generate.py number")
        sys.exit(1)

    with open('data/skills.txt', 'r') as f:
        skills = f.readlines()
    skills = [skill.strip() for skill in skills]

    with open('data/soft_skills.txt', 'r') as f:
        soft_skills = f.readlines()
    soft_skills = [skill.strip() for skill in soft_skills]

    with open('data/education.txt', 'r') as f:
        education = f.readlines()
    education = [edu.strip() for edu in education]

    number = int(sys.argv[1])
    for i in range(number):
        d = main(skills, soft_skills, education, f"generate_resume{i}.txt")
        write_to_csv(d, 'dataset.csv')