#!/bin/zsh
# One sitting: a fresh folder with only the data, the prompt and the settings; Claude Code headless; full transcript kept.
# usage: tools/run_sitting.sh s1
set -e
N=$1; EXP=${0:A:h:h}; RUNS=$HOME/Documents/Projects/tess-runs; D=$RUNS/$N
[[ -e $D ]] && { echo "$D exists"; exit 1; }
mkdir -p $D/.claude $D/plots; cp -R $EXP/task/data $D/data; cp $EXP/task/settings.json $D/.claude/settings.json
cd $D
date -u +%Y-%m-%dT%H:%M:%SZ > $EXP/results/$N.start
/usr/bin/time -p claude -p "$(cat $EXP/task/PROMPT.md)" --model claude-opus-5-5 --effort high \
  --setting-sources project --strict-mcp-config --allowedTools Bash Read Write Edit Glob Grep \
  --disallowedTools WebFetch WebSearch 'Bash(curl:*)' 'Bash(wget:*)' 'Bash(pip:*)' 'Bash(pip3:*)' 'Bash(git clone:*)' \
  --output-format stream-json --verbose > $EXP/results/$N.stream.jsonl 2> $EXP/results/$N.stderr
date -u +%Y-%m-%dT%H:%M:%SZ > $EXP/results/$N.end
mkdir -p $EXP/results/$N && cp -R $D/results.json $D/NOTES.md $D/plots $D/*.py $EXP/results/$N/ 2>/dev/null || true
echo "sitting $N done"
