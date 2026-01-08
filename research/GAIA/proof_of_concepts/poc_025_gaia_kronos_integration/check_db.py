#!/usr/bin/env python3
"""Debug script to check database content."""

import sqlite3
from pathlib import Path

db_path = Path('./kronos_qa/knowledge/graph_knowledge.db')
if not db_path.exists():
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(f"Tables: {tables}")

# Count nodes
c.execute("SELECT count(*) FROM nodes")
print(f"Total nodes: {c.fetchone()[0]}")

# Find PAC definitions
c.execute("SELECT id, content FROM nodes WHERE content LIKE '%PAC:%' OR content LIKE '%PAC (%' LIMIT 10")
results = c.fetchall()
print(f"\nFound {len(results)} PAC-related nodes:")
for r in results:
    print(f"  ID: {r[0][:20]}...")
    print(f"  Content: {r[1][:120]}...")
    print()

conn.close()
