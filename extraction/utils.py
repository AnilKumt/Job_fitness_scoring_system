import csv
import os
import re
from dateutil import parser

def calculate_experience(start_date, end_date):
    start_date = parser.parse(start_date)
    end_date = parser.parse(end_date)

    yr = end_date.year - start_date.year
    months = end_date.month - start_date.month

    total_months = yr * 12 + months
    return total_months / 12.0

def extract_experience(text):
    # May. 2002 - June. 2003 or May 2002 - June 2003
    pattern = r'(\b[A-Za-z]+\.*\s\d{4}\b)\s*-\s*(\b[A-Za-z]+\.*\s\d{4}\b)'
    exp = []
    matches = re.findall(pattern, text)

    for match in matches:
        start_date = match[0]
        end_date = match[1]

        exp_yrs = calculate_experience(start_date, end_date)
        exp.append(exp_yrs)

    return round(sum(exp), 3)

def write_to_csv(data, output_file):
    file_exists = os.path.isfile(output_file)

    if os.path.isfile(output_file):
        with open(output_file, mode="r") as f:
            rows = list(csv.reader(f))
            last_id = int(rows[-1][0]) if len(rows) > 1 else 0
        data['resume_id'] = last_id + 1
    else:
        data['resume_id'] = 0

    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            if output_file != 'dataset.csv':
                writer.writerow(['id', 'skills', 'experience_years', 'education_degree', 'label', 'output_file'])
            else:
                writer.writerow(['resume_id', 'skills', 'experience_years', 'education_degree', 'label', 'output_file'])

        writer.writerow([data['resume_id'],
                        ', '.join(data['skills']),
                        data['experience'],
                        data['highest_degree'],
                        0,
                        data['output_file']])

def custom_tokenizer(nlp):
    special_cases = {"Master's": [{"ORTH": "Master's"}],
                     "Bachelor's": [{"ORTH": "Bachelor's"}]}
    nlp.tokenizer.add_special_case("Master's", special_cases["Master's"])
    nlp.tokenizer.add_special_case("Bachelor's", special_cases["Bachelor's"])
    return nlp