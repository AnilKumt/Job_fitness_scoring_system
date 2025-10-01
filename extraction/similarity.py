from sentence_transformers import SentenceTransformer, util
import extract as e
import sys

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def dict_to_text_skills_and_degree(d):
    skills = d.get('skills', [])
    degree = d.get('highest_degree', '')

    # If skills and degree are both missing, return an empty string
    if not skills and not degree:
        return ""

    parts = []
    if skills:
        skills_str = ', '.join(skills).lower()
        parts.append(f"Skills: {skills_str}.")
    if degree:
        parts.append(f"Degree: {degree.lower()}.")

    return ' '.join(parts)


def compute_similarity(d1, d2):
    """
    Compute cosine similarity between d1 and d2 based on skills and degree only.
    """
    text1 = dict_to_text_skills_and_degree(d1)
    text2 = dict_to_text_skills_and_degree(d2)

    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return round(similarity, 3)

# --- Main execution ---

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python similarity.py <resume_txt_file> <jd_txt_file>")
        sys.exit(1)

    resume = sys.argv[1]
    jd = sys.argv[2]
    # Load data
    d1 = e.main(resume)
    d2 = e.main(jd)

    # Compute and print similarity
    sim_score = compute_similarity(d1, d2)
    print(d1)
    print(d2)
    print(f"Skills + Degree Similarity Score: {sim_score}")
