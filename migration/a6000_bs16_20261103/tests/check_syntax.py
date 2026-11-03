from pathlib import Path
p=Path(__file__).resolve().parents[1]
files=list(p.rglob('*.py'))
for f in files:compile(f.read_bytes(),str(f),'exec')
print('PYTHON_SYNTAX_PASS',len(files),'files; no imports/cache')
