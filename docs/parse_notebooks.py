import json, re, os, shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def convert_links_to_sphinx(content):
    # Regular expression to find Markdown links to Python files
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+\.py)\)')
    
    # Replace Markdown links with Sphinx directives
    def replace_link(match):
        text = match.group(1)  # name of the link
        link = match.group(2)  # path to the Python file
        module_doc_path = link.replace('/', '.').replace('.py', '').lstrip('.')
        return f':py:mod:`{module_doc_path}`'
    
    return pattern.sub(replace_link, content)

def process_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook_content = json.load(f)
    
    # Process each cell in the notebook
    for cell in notebook_content['cells']:
        if cell['cell_type'] == 'markdown':
            cell['source'] = [convert_links_to_sphinx(line) for line in cell['source']]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2)

def copy_and_parse_notebooks_to_docs(
    source_dir=os.path.join(project_root, 'getting_started', 'tutorials'),
    dest_dir=os.path.join(project_root, 'docs', 'tutorials')
    ):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    
    # Process each notebook in the destination directory
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.endswith('.ipynb'):
                process_notebook(os.path.join(root, f))