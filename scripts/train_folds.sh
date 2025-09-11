
#!/usr/bin/env bash
python -m src.main train --data_dir "${1:-./data}" --out_dir "${2:-./outputs}"
