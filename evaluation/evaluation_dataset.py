
EVALUATION_DATASET = [
    # RAG fundamentals
    {
        "question": "What are embeddings in RAG?",
        "expected_chunks": [("rag_basics.txt", 0)],
        "answerable": True,
    },
    {
        "question": "Does RAG require fine-tuning?",
        "expected_chunks": [("rag_basics.txt", 0)],
        "answerable": True,
    },
    {
        "question": "What is source traceability?",
        "expected_chunks": [
            ("rag_basics.txt", 0),
            ("rag_basics.txt", 1),
        ],
        "answerable": True,
    },
    {
        "question": "What does chunk overlap do?",
        "expected_chunks": [("rag_basics.txt", 1)],
        "answerable": True,
    },

    # Machine learning
    {
        "question": "What is supervised learning?",
        "expected_chunks": [("machine_learning.txt", 0)],
        "answerable": True,
    },
    {
        "question": "What is unsupervised learning?",
        "expected_chunks": [("machine_learning.txt", 1)],
        "answerable": True,
    },
    {
        "question": "What is the difference between classification and regression?",
        "expected_chunks": [("machine_learning.txt", 2)],
        "answerable": True,
    },
    {
        "question": "What is K-Means clustering?",
        "expected_chunks": [("machine_learning.txt", 2), ("machine_learning.txt", 3)],
        "answerable": True,
    },
    {
        "question": "What is overfitting?",
        "expected_chunks": [("machine_learning.txt", 4)],
        "answerable": True,
    },

    # Deep learning
    {
        "question": "What is an artificial neural network?",
        "expected_chunks": [("deep_learning.txt", 0)],
        "answerable": True,
    },
    {
        "question": "What is backpropagation?",
        "expected_chunks": [("deep_learning.txt", 2)],
        "answerable": True,
    },
    {
        "question": "What is a CNN used for?",
        "expected_chunks": [("deep_learning.txt", 3)],
        "answerable": True,
    },
    {
        "question": "What is self-attention in transformers?",
        "expected_chunks": [("deep_learning.txt", 4)],
        "answerable": True,
    },

    # LLM fundamentals
    {
        "question": "What is a context window?",
        "expected_chunks": [("llm_fundamentals.txt", 1)],
        "answerable": True,
    },
    {
        "question": "What is LoRA fine-tuning?",
        "expected_chunks": [("llm_fundamentals.txt", 4), ("llm_fundamentals.txt", 5)],
        "answerable": True,
    },
    {
        "question": "What is quantization?",
        "expected_chunks": [("llm_fundamentals.txt", 5), ("llm_fundamentals.txt", 6)],
        "answerable": True,
    },
    {
        "question": "What causes hallucinations in LLMs?",
        "expected_chunks": [("llm_fundamentals.txt", 7)],
        "answerable": True,
    },
    {
        "question": "What is the difference between RAG and fine-tuning?",
        "expected_chunks": [("llm_fundamentals.txt", 8), ("llm_fundamentals.txt", 9)],
        "answerable": True,
    },

    # Databases and SQL
    {
        "question": "What is the difference between a primary key and a foreign key?",
        "expected_chunks": [("databases_sql.txt", 3), ("databases_sql.txt", 4)],
        "answerable": True,
    },
    {
        "question": "What is the difference between INNER JOIN and LEFT JOIN?",
        "expected_chunks": [("databases_sql.txt", 5), ("databases_sql.txt", 6)],
        "answerable": True,
    },
    {
        "question": "What is the difference between WHERE and HAVING?",
        "expected_chunks": [("databases_sql.txt", 7)],
        "answerable": True,
    },
    {
        "question": "What is database normalization?",
        "expected_chunks": [("databases_sql.txt", 8)],
        "answerable": True,
    },
    {
        "question": "What are the four ACID properties?",
        "expected_chunks": [("databases_sql.txt", 9), ("databases_sql.txt", 10)],
        "answerable": True,
    },

    # Unanswerable questions
    {
        "question": "What is the capital of France?",
        "expected_chunks": [],
        "answerable": False,
    },
    {
        "question": "Who invented the telephone?",
        "expected_chunks": [],
        "answerable": False,
    },
    {
        "question": "What is the current stock price of NVIDIA?",
        "expected_chunks": [],
        "answerable": False,
    },
]
