# Remote helpers for ember3_scale160m (node1 / CT103, everything under /data/ember3).
# Usage: .\remote.ps1 push|prep|preflight|matrix|pull|status
param([Parameter(Mandatory = $true)][string]$cmd)

$scripts = $PSScriptRoot
$results = Join-Path $PSScriptRoot "..\results"
$PY = "/data/ember3/venv/bin/python"

switch ($cmd) {
    "push" {
        scp "$scripts\prep_data.py" "$scripts\run_scale160m.py" node1:/tmp/
        ssh node1 "pct push 103 /tmp/prep_data.py /data/ember3/prep_data.py; pct push 103 /tmp/run_scale160m.py /data/ember3/run_scale160m.py; rm /tmp/prep_data.py /tmp/run_scale160m.py"
    }
    "prep" {
        ssh node1 "pct exec 103 -- bash -lc 'cd /data/ember3 && nohup $PY prep_data.py > prep.log 2>&1 & echo started'"
    }
    "preflight" {
        ssh node1 "pct exec 103 -- bash -lc 'cd /data/ember3 && $PY run_scale160m.py --arm err --chunks 200'"
    }
    "matrix" {
        # Sequential on the single GPU; frozen_ref first (analysis reference).
        $seq = "$PY run_scale160m.py --arm frozen_ref && $PY run_scale160m.py --arm none && $PY run_scale160m.py --arm err && $PY run_scale160m.py --arm err --rep 1 && $PY run_scale160m.py --arm ent && $PY run_scale160m.py --arm rand --seed 0 && $PY run_scale160m.py --arm rand --seed 1 && $PY run_scale160m.py --arm rand --seed 2"
        ssh node1 "pct exec 103 -- bash -lc 'cd /data/ember3 && nohup bash -c `"$seq`" > matrix.log 2>&1 & echo started'"
    }
    "status" {
        ssh node1 "pct exec 103 -- bash -lc 'tail -5 /data/ember3/matrix.log 2>/dev/null; tail -3 /data/ember3/prep.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'"
    }
    "pull" {
        ssh node1 "pct exec 103 -- tar czf /tmp/e3r.tgz -C /data/ember3 results; pct pull 103 /tmp/e3r.tgz /tmp/e3r.tgz"
        scp node1:/tmp/e3r.tgz "$env:TEMP\e3r.tgz"
        if (-not (Test-Path $results)) { New-Item -ItemType Directory $results | Out-Null }
        tar xzf "$env:TEMP\e3r.tgz" -C $results --strip-components=1
        ssh node1 "rm -f /tmp/e3r.tgz"
        Write-Host "results pulled to $results"
    }
}
