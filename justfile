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

show-prompt:
    source .venv/bin/activate && comma-lab show-prompt top
