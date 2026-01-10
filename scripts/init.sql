-- 1. Enable pgvector extension (Required for vector storage)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Initialize the main table with a vector column
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(768), -- The 768-dim vector column for AI analysis
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Enable logical replication for this table (required for CDC)
ALTER TABLE articles REPLICA IDENTITY FULL;

-- 3. Create the History Log (CDC Log)
-- This table automatically archives every version of a row to build trajectories for the Neural ODE
CREATE TABLE IF NOT EXISTS cdc_log (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    row_data JSONB, -- Stores the full row state including the vector
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast history retrieval by the Auditor
CREATE INDEX IF NOT EXISTS idx_cdc_log_record ON cdc_log(table_name, record_id);
CREATE INDEX IF NOT EXISTS idx_cdc_log_time ON cdc_log(changed_at);

-- 4. Create the History Trigger
-- Automatically populates cdc_log whenever articles are modified
CREATE OR REPLACE FUNCTION log_cdc_event() RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO cdc_log (table_name, record_id, operation, row_data, changed_at)
    VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(NEW)::jsonb, NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS articles_cdc_trigger ON articles;
CREATE TRIGGER articles_cdc_trigger
AFTER INSERT OR UPDATE ON articles
FOR EACH ROW EXECUTE FUNCTION log_cdc_event();

-- 5. Create publication if it doesn't exist
-- Note: This often requires superuser privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'auditor_pub') THEN
        CREATE PUBLICATION auditor_pub FOR TABLE articles;
    END IF;
END
$$;

-- 6. Insert some dummy data with a Real Vector (Zero-padded for demo)
-- pgvector format is simply '[1,2,3...]'
INSERT INTO articles (id, title, content, embedding) VALUES
('1', 'Introduction to Vector Search', 'Vector search allows for semantic similarity matching...', (SELECT array_to_vector(array_fill(0.1::float4, ARRAY[768])))),
('2', 'Consistency in Distributed Systems', 'CAP theorem states that...', (SELECT array_to_vector(array_fill(0.2::float4, ARRAY[768]))))
ON CONFLICT (id) DO NOTHING;