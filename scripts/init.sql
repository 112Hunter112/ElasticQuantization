-- Initialize the database schema
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Enable logical replication for this table (required for CDC)
ALTER TABLE articles REPLICA IDENTITY FULL;

-- Create publication if it doesn't exist
-- Note: This often requires superuser privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'auditor_pub') THEN
        CREATE PUBLICATION auditor_pub FOR TABLE articles;
    END IF;
END
$$;

-- Insert some dummy data
INSERT INTO articles (id, title, content) VALUES
('1', 'Introduction to Vector Search', 'Vector search allows for semantic similarity matching...'),
('2', 'Consistency in Distributed Systems', 'CAP theorem states that...');
