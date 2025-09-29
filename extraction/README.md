# What Does This Do ?
- extract_text_from_pdf.py takes input of a pdf file and converts it into a a txt file and saves it in output folder with same name.
- extract.py takes that txt file in output folder and prints a dictionary of the following
1. name
2. education
3. certifications
4. skills
5. job_role
- These are the ones till now, will be updated as required.

# Usage
- from extraction directory
`
python .\extract_text_from_pdf.py .\data\resume.pdf
`
`
python .\extract.py .\output\resume.txt
`

# How it works ?
- PyMuPDF is used to convert pdf files to text files and they are stored in output directory.
- extract.py takes all the lists from data like names, skills, education, certifications and job_roles for pattern matching. No ML or model is used, just normal pattern matching.