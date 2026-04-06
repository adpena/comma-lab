default:
    @just --list

start:
    bash start.sh

bootstrap:
    bash scripts/bootstrap.sh

doctor:
    source .venv/bin/activate && comma-lab doctor

status:
    source .venv/bin/activate && comma-lab status

reproduce-exact-current:
    source .venv/bin/activate && comma-lab eval-submission exact_current --device cpu --report-copy reports/raw/exact_current-current_workflow-cpu-report.txt | tee reports/raw/exact_current-current_workflow-cpu-summary.json

reproduce-robust-current:
    source .venv/bin/activate && comma-lab eval-submission robust_current --package --device cpu --report-copy reports/raw/robust_current-current_workflow-cpu-report.txt | tee reports/raw/robust_current-current_workflow-cpu-summary.json

rebuild-site:
    python3 reports/graphs/refresh_site.py

show-prompt:
    source .venv/bin/activate && comma-lab show-prompt top
