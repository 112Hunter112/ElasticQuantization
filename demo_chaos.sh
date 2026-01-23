#!/bin/bash

# demo_chaos.sh
# Demonstrates the Auto-Healing capabilities of the Consistency Auditor.

# Configuration
ES_URL="http://localhost:9200"
INDEX="documents"
ID="chaos-demo-1"

echo "=== ElasticQuantization Chaos Demo ==="
echo "Ensuring clean state..."

# 1. Insert a valid record into Postgres
echo "[1] Inserting legitimate record into PostgreSQL..."
docker compose exec -T postgres psql -U postgres -d consistency_db -c "
INSERT INTO articles (id, title, content) 
VALUES ('$ID', 'Original Title', 'This is the original content.') 
ON CONFLICT (id) DO UPDATE SET content = 'This is the original content.';
"

echo "Waiting for Auditor to sync..."
sleep 2

# 2. Verify ES has the data (Optional check)
# curl -s "$ES_URL/$INDEX/_doc/$ID" | grep "Original Title"

# 3. Create Discrepancy: Manually corrupt Elasticsearch data
echo "[2] SABOTAGE: Manually corrupting Elasticsearch data..."
curl -X PUT "$ES_URL/$INDEX/_doc/$ID?refresh=true" -H 'Content-Type: application/json' -d '{
    "title": "HACKED TITLE",
    "content": "This data has been corrupted by the chaos script!",
    "vector_quantized": "AAAA" 
}'
echo ""
echo "Discrepancy created! DB has 'Original Title', ES has 'HACKED TITLE'."

# 4. Trigger the Auditor (via CDC)
# Since the auditor listens to CDC, we trigger a harmless update in DB to force a check.
echo "[3] Triggering Auditor check via DB update..."
docker compose exec -T postgres psql -U postgres -d consistency_db -c "
UPDATE articles SET updated_at = NOW() WHERE id = '$ID';
"

echo "[4] Watching logs for healing..."
echo "---------------------------------------------------"
# Follow logs specifically for our ID
docker compose logs -f auditor | grep --line-buffered "$ID"
