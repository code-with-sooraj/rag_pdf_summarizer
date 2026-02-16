import os

DOC_DIR = "documents"

os.makedirs(DOC_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file):

    file_path = os.path.join(DOC_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path
