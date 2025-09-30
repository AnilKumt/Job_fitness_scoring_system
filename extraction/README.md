# What Does This Do ?
- extract_text_from_pdf.py takes input of a pdf file and converts it into a a txt file and saves it in output folder with same name.
- extract.py takes that txt file in output folder and prints into dataset.csv
1. resume_id
2. skills (Ex: "sql, python, nlp")
3. experience_years
4. education_degree (Highest, liberty taken, degrees will be from higest to lowest acc. resume format)
5. label (default 0)
6. Output file
- batch_run.py works for a batch of pdf.
- These are the ones till now, will be updated as required.

# Usage
- Updated one, from extraction directory
```bash
python .\main.py .\data\resume.pdf
```
- This does the work of the below two.
- from extraction directory
```bash
python .\extract_text_from_pdf.py .\data\resume.pdf
```
```bash
python .\extract.py .\output\resume.txt
```
- For batch running
```bash
python .\batch_run.py
```

# How it works ?
- PyMuPDF is used to convert pdf files to text files and they are stored in output directory.
- extract.py takes all the lists from data like skills, education for pattern matching. Regex is used for experience calculation. No ML or model is used, just normal pattern matching.

# To Do:
1. Should add more rows into the dataset.
2. Should manually add skills and education.
3. Check for edge cases and change accordingly.
