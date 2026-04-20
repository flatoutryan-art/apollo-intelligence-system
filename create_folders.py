import os
path = os.path.join(os.getcwd(), "export")
os.makedirs(path, exist_ok=True)
print(f"Folder 'export' should now exist at: {path}")