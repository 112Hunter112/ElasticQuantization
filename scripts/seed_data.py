import requests
import psycopg2
import random
from typing import List, Dict
from datetime import datetime, timedelta
import json
from tqdm import tqdm  # <--- NEW IMPORT

# Configuration
ML_URL = "http://localhost:5000/generate_embedding"
DB_DSN = "postgresql://postgres:password@localhost:5432/consistency_db"

# Enhanced topic structure with subtopics and variations
TOPIC_CLUSTERS = {
    "distributed_systems": {
        "core_concepts": [
            "Database consistency is critical for distributed systems",
            "CAP theorem governs distributed database design",
            "Consensus algorithms enable distributed coordination",
            "Eventual consistency provides availability guarantees",
            "Partition tolerance requires careful trade-offs"
        ],
        "technologies": [
            "Raft consensus ensures leader election",
            "Paxos provides distributed agreement",
            "CRDT enables conflict-free replication",
            "Vector clocks track causality",
            "Gossip protocols spread information efficiently"
        ]
    },
    "machine_learning": {
        "neural_networks": [
            "Neural ODEs model continuous time dynamics",
            "Residual networks enable deep learning",
            "Attention mechanisms improve sequence modeling",
            "Graph neural networks process relational data",
            "Transformers revolutionized NLP architectures"
        ],
        "training": [
            "Gradient descent optimizes model parameters",
            "Backpropagation computes gradients efficiently",
            "Batch normalization stabilizes training",
            "Dropout prevents overfitting",
            "Learning rate scheduling improves convergence"
        ]
    },
    "search_retrieval": {
        "vector_search": [
            "Vector search enables semantic retrieval",
            "Approximate nearest neighbors scale to billions",
            "HNSW provides efficient graph-based search",
            "Product quantization compresses vectors",
            "Cosine similarity measures semantic distance"
        ],
        "traditional": [
            "Elasticsearch uses inverted indices for text search",
            "BM25 ranks documents by relevance",
            "TF-IDF weights term importance",
            "Fuzzy matching handles typos",
            "Boolean queries combine search criteria"
        ]
    },
    "infrastructure": {
        "orchestration": [
            "Kubernetes orchestration manages container lifecycles",
            "Docker containers provide isolation",
            "Service mesh handles microservice communication",
            "Load balancers distribute traffic",
            "Auto-scaling responds to demand"
        ],
        "databases": [
            "Postgres provides ACID compliance for transactions",
            "Redis offers in-memory caching",
            "MongoDB stores flexible document schemas",
            "Cassandra scales horizontally",
            "TimescaleDB optimizes time-series data"
        ]
    },
    "programming": {
        "languages": [
            "Golang is excellent for high-concurrency systems",
            "Python simplifies rapid development",
            "Rust ensures memory safety",
            "TypeScript adds type safety to JavaScript",
            "Java provides enterprise-grade reliability"
        ],
        "patterns": [
            "Dependency injection improves testability",
            "Observer pattern enables event-driven design",
            "Factory pattern abstracts object creation",
            "Singleton pattern ensures single instance",
            "Repository pattern separates data access"
        ]
    }
}

# Templates for generating varied content
CONTENT_TEMPLATES = [
    "{statement}. This is fundamental to understanding {domain}.",
    "In the context of {domain}, {statement}.",
    "{statement}, which makes it essential for {application}.",
    "Recent advances show that {statement}.",
    "Practitioners have found that {statement} in production environments.",
    "{statement}. This principle underlies many {domain} implementations.",
    "When building {application}, remember that {statement}.",
    "The key insight is that {statement}.",
    "{statement}. This has significant implications for {domain} architecture.",
    "Research indicates that {statement}, particularly in {application}."
]

DOMAINS = [
    "modern software architecture",
    "scalable systems",
    "production deployments",
    "enterprise applications",
    "cloud-native infrastructure",
    "data-intensive systems",
    "real-time processing",
    "distributed computing"
]

APPLICATIONS = [
    "web services",
    "data pipelines",
    "recommendation systems",
    "search engines",
    "analytics platforms",
    "microservices",
    "API gateways",
    "streaming applications"
]

def get_vector(text: str) -> List[float]:
    """Get embedding from ML service"""
    try:
        # Increased timeout for CPU-based inference
        resp = requests.post(ML_URL, json={"text": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()['embedding']
    except Exception as e:
        print(f"ML Service Error: {e}")
        return None

def generate_document(doc_id: int) -> Dict:
    """Generate a diverse document with metadata"""
    
    # Select random topic cluster and category
    cluster = random.choice(list(TOPIC_CLUSTERS.keys()))
    category = random.choice(list(TOPIC_CLUSTERS[cluster].keys()))
    statement = random.choice(TOPIC_CLUSTERS[cluster][category])
    
    # Generate varied content using templates
    template = random.choice(CONTENT_TEMPLATES)
    content = template.format(
        statement=statement,
        domain=random.choice(DOMAINS),
        application=random.choice(APPLICATIONS)
    )
    
    # Add additional sentences for complexity (randomly)
    if random.random() > 0.5:
        extra_statement = random.choice(TOPIC_CLUSTERS[cluster][category])
        content += f" Additionally, {extra_statement.lower()}."
    
    # Generate realistic title
    title_patterns = [
        f"Understanding {category.replace('_', ' ').title()} in {cluster.replace('_', ' ').title()}",
        f"A Guide to {category.replace('_', ' ').title()}",
        f"{cluster.replace('_', ' ').title()}: {category.replace('_', ' ').title()} Best Practices",
        f"Article {doc_id}: {category.replace('_', ' ').title()}"
    ]
    title = random.choice(title_patterns)
    
    # Add metadata
    metadata = {
        "cluster": cluster,
        "category": category,
        "doc_type": random.choice(["article", "tutorial", "reference", "guide"]),
        "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
        "created_at": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
        "tags": [cluster, category, random.choice(["production", "research", "practical"])]
    }
    
    return {
        "id": str(doc_id),
        "title": title,
        "content": content,
        "metadata": json.dumps(metadata)
    }

def generate_cross_cluster_documents(num_docs: int) -> List[Dict]:
    """Generate documents that span multiple clusters for richer relationships"""
    docs = []
    
    for i in range(num_docs):
        # Mix concepts from 2-3 clusters
        num_clusters = random.randint(2, 3)
        selected_clusters = random.sample(list(TOPIC_CLUSTERS.keys()), num_clusters)
        
        statements = []
        for cluster in selected_clusters:
            category = random.choice(list(TOPIC_CLUSTERS[cluster].keys()))
            statement = random.choice(TOPIC_CLUSTERS[cluster][category])
            statements.append(statement)
        
        content = " ".join(statements) + f" These concepts work together in {random.choice(APPLICATIONS)}."
        
        doc = {
            "id": str(1000 + i),  # Different ID range for cross-cluster docs
            "title": f"Integration Study {i}: {' & '.join(selected_clusters)}",
            "content": content,
            "metadata": json.dumps({
                "clusters": selected_clusters,
                "doc_type": "integration",
                "cross_domain": True
            })
        }
        docs.append(doc)
    
    return docs

def seed_database(num_regular: int = 100, num_cross_cluster: int = 20, batch_size: int = 10):
    """Seed database with diverse documents"""
    
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    
    # Create table if needed (with metadata column)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            embedding vector,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    print(f"🌱 Seeding {num_regular} regular documents + {num_cross_cluster} cross-cluster documents...")
    
    # Generate regular documents
    # Using tqdm for progress bar
    for i in tqdm(range(1, num_regular + 1), desc="Seeding Regular Docs", unit="doc"):
        doc = generate_document(i)
        vector = get_vector(doc["content"])
        
        if vector:
            sql = """
            INSERT INTO articles (id, title, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE 
            SET content=EXCLUDED.content, 
                embedding=EXCLUDED.embedding,
                metadata=EXCLUDED.metadata;
            """
            cur.execute(sql, (doc["id"], doc["title"], doc["content"], vector, doc["metadata"]))
            
            # Commit batch to keep transaction logs manageable
            if i % batch_size == 0:
                conn.commit()
    
    # Ensure final batch is committed
    conn.commit()

    # Generate cross-cluster documents
    cross_docs = generate_cross_cluster_documents(num_cross_cluster)
    
    # Using tqdm for progress bar
    for i, doc in enumerate(tqdm(cross_docs, desc="Seeding Cross-Cluster", unit="doc"), 1):
        vector = get_vector(doc["content"])
        
        if vector:
            sql = """
            INSERT INTO articles (id, title, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE 
            SET content=EXCLUDED.content, 
                embedding=EXCLUDED.embedding,
                metadata=EXCLUDED.metadata;
            """
            cur.execute(sql, (doc["id"], doc["title"], doc["content"], vector, doc["metadata"]))
    
    conn.commit()
    
    # Print statistics
    cur.execute("SELECT COUNT(*) FROM articles")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT metadata->>'cluster') FROM articles WHERE metadata->>'cluster' IS NOT NULL")
    unique_clusters = cur.fetchone()[0]
    
    conn.close()
    
    print(f"✅ Seeding complete!")
    print(f"   Total documents: {total}")
    print(f"   Unique clusters: {unique_clusters}")
    print(f"   Cross-cluster docs: {num_cross_cluster}")

if __name__ == "__main__":
    # Customize these parameters
    seed_database(
        num_regular=2000,
        num_cross_cluster=200,
        batch_size=50
    )