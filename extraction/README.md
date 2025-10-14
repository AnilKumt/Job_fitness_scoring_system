# What does each File do ?
## batch_run_jd.py
- Run this file and select the folder which contain job description
- Output will be updated in output/jd as .txt file, and in dataset_jd.csv
```bash
python ./batch_run_jd.py
```
## batch_run_resume.py
- Run this file and select the folder which contain resume.
- Output will be updated in output/resume as .txt file, and in dataset.csv
```bash
python ./batch_run_resume.py
```
## extract.py
- Extracts the .txt and saves into given output csv
```bash
python .\extract.py .\output\resume\resume.txt ./dataset.csv
```
## extract_text_from_pdf.py
- Extracts the .txt from the given pdf file
- Format, extract_text_from_pdf.py <resume.pdf> <output_csv>
```bash
python .\extract_text_from_pdf.py .\data\resume.pdf ./dataset.csv
```
## main.py
- Does the work of both extract.py and extract_text_from_pdf.py, rather than running two programs one after another.
```bash
python ./main.py ./data/resume.pdf ./dataset.csv
```
## similarity.py
- Compares resume and jd.
```bash
python ./similarity.py output/resume/sample.txt output/jd/jd.txt
```
## generate.py
- Generates given number of rows in the dataset.csv file
```bash
python ./generate.py 100
```
- This generates 100 lines of new random data in dataset.csv file



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
- similarity.py takes jd and resume and checks their similarity.
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
python .\extract.py .\output\resume\resume.txt
```
- For batch running
```bash
python .\batch_run.py
```
- For similarity.py
```bash
python .\similarity.py .\output\resume\resume.txt .\output\jd\jd.txt
```

# How it works ?
- PyMuPDF is used to convert pdf files to text files and they are stored in output directory.
- extract.py takes all the lists from data like skills, education for pattern matching. Regex is used for experience calculation. No ML or model is used, just normal pattern matching.
- similarity.py has a simple checking for the skills and other using transformer model.

# To Do:
1. Should add more rows into the dataset.
2. Should manually add skills and education.
3. Check for edge cases and change accordingly.
