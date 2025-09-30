import pymupdf
import sys
import os

def extract_text_from_pdf(pdf_file):
    doc = pymupdf.open(pdf_file)
    pdf_text = []
    for page in doc:
        content = page.get_text()
        pdf_text.append(content)

    return pdf_text

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("python extract_text_from_pdf.py <pdf_file> <output_csv>")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_csv = sys.argv[2]
    extracted_text = extract_text_from_pdf(input_pdf)
    if output_csv == 'dataset.csv':
        output_txt = os.path.join("output", "resume", os.path.basename(input_pdf.rsplit('.', 1)[0]) + ".txt") 
    else:
        output_txt = os.path.join("output", "jd", os.path.basename(input_pdf.rsplit('.', 1)[0]) + ".txt") 

    with open(output_txt, 'w', encoding='utf-8') as f:
        for text in extracted_text:
            f.write(text)





