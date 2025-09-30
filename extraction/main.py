import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Error")
        return
    
    path = sys.argv[1]

    if not path.lower().endswith(".pdf"):
        print("Error, pdf files")
        return
    
    output_txt = os.path.join("output", os.path.basename(path.rsplit('.', 1)[0]) + ".txt") 

    result1 = subprocess.run(["python", "extract_text_from_pdf.py", path])
    if result1.returncode != 0:
        print("Failed pdf extract")
        return
    
    result2 = subprocess.run(["python", "extract.py", output_txt])
    if result2.returncode != 0:
        print("Failed to write to output")
        return
    
if __name__ == "__main__":
    main()
