import os
import subprocess
from tkinter import Tk, filedialog

def select_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Resume folder")
    return folder_path

def main(folder_path):
    if not folder_path:
        return 
    
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder_path, fname)
            subprocess.run(["python", "main.py", path])

if __name__ == "__main__":
    folder = select_folder()
    main(folder)