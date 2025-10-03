import sys
import os
import subprocess

def main():
    if len(sys.argv) < 3:
        print("path, output file")
        return
    
    path = sys.argv[1]
    output_file = sys.argv[2]

    if not path.lower().endswith(".pdf"):
        print("Error, pdf files")
        return
    
    if output_file == 'dataset.csv':
        output_txt = os.path.join("output", "resume", os.path.basename(path.rsplit('.', 1)[0]) + ".txt") 
    else:
        output_txt = os.path.join("output", "jd", os.path.basename(path.rsplit('.', 1)[0]) + ".txt") 

    result1 = subprocess.run(["python", "extract_text_from_pdf.py", path, output_file])
    if result1.returncode != 0:
        print("Failed pdf extract")
        return
    
    result2 = subprocess.run(["python", "extract.py", output_txt, output_file])
    if result2.returncode != 0:
        print("Failed to write to output")
        return
    
if __name__ == "__main__":
    main()
