#!/usr/bin/env bash
_teacher_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TEACHER_DISTILLATION_ROOT="$_teacher_project_root"
export REACT_PYTHON="${REACT_PYTHON:-$_teacher_project_root/.venv/bin/python}"
export REACT_DATA_DIR="${REACT_DATA_DIR:-$_teacher_project_root/data}"
export MAM_REACTOR_ROOT="${MAM_REACTOR_ROOT:-$_teacher_project_root/mam_reactor}"
export PYTHONPATH="$_teacher_project_root:$_teacher_project_root/mam_reactor${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$_teacher_project_root/.mplconfig}"
unset _teacher_project_root
