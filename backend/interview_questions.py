"""
Role-based Interview Questions Database with AI-driven Dynamic Generation
"""
import json
import random
from processor import client, MODEL
from gemini_integration import gemini_api

INTERVIEW_QUESTIONS = {
    'software_developer': {
        'role': 'Software Developer',
        'description': 'Backend and Full-Stack Development',
        'category': 'job',
        'self_introduction': {
            'question': 'Tell me about yourself',
            'guidance': 'Cover: name, background, skills, experience, why you want this role (60-90 seconds)',
            'tips': [
                'Start with brief personal intro',
                'Mention relevant education/certifications',
                'Highlight key technical skills',
                'Briefly describe relevant work experience',
                'Explain why you are interested in this role'
            ]
        },
        'beginner': [
            {'id': 1, 'question': 'What do you understand by OOP (Object-Oriented Programming)?', 'expected_keywords': ['classes', 'objects', 'inheritance', 'polymorphism', 'encapsulation']},
            {'id': 2, 'question': 'Explain the difference between ArrayList and LinkedList in Java', 'expected_keywords': ['time complexity', 'memory', 'insertion', 'deletion']},
            {'id': 3, 'question': 'What are the SOLID principles?', 'expected_keywords': ['Single Responsibility', 'Open/Closed', 'Liskov', 'Interface Segregation', 'Dependency']},
            {'id': 4, 'question': 'How does garbage collection work in Java/Python?', 'expected_keywords': ['memory management', 'heap', 'automatic cleanup']},
            {'id': 5, 'question': 'What is a REST API and its main principles?', 'expected_keywords': ['HTTP', 'stateless', 'resources', 'methods']},
            {'id': 11, 'question': 'What is the difference between an interface and an abstract class?', 'expected_keywords': ['contract', 'implementation', 'multiple inheritance']},
            {'id': 12, 'question': 'Explain the concept of "Pass by Value" vs "Pass by Reference".', 'expected_keywords': ['memory', 'pointer', 'copy']},
            {'id': 13, 'question': 'What are the primary data structures in computer science?', 'expected_keywords': ['Array', 'Stack', 'Queue', 'Linked List', 'Tree']},
            {'id': 14, 'question': 'What is the purpose of an Index in a database?', 'expected_keywords': ['search optimization', 'B-Tree', 'performance']},
            {'id': 15, 'question': 'Explain the difference between SQL and NoSQL databases.', 'expected_keywords': ['relational', 'schema-less', 'scaling']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'Design a URL shortening service like bit.ly', 'expected_keywords': ['hashing', 'scalability', 'base62', 'short URL']},
            {'id': 7, 'question': 'How would you implement caching in a high-traffic application?', 'expected_keywords': ['Redis', 'Memcached', 'LRU', 'eviction']},
            {'id': 8, 'question': 'Explain microservices architecture and its trade-offs.', 'expected_keywords': ['decoupling', 'service discovery', 'API gateway']},
            {'id': 9, 'question': 'How do you handle database transactions and ACID properties?', 'expected_keywords': ['Atomicity', 'Consistency', 'Isolation', 'Durability']},
            {'id': 10, 'question': 'Design a real-time notification system.', 'expected_keywords': ['WebSockets', 'Pub/Sub', 'Message Queues']},
            {'id': 16, 'question': 'How do you handle race conditions in multi-threaded applications?', 'expected_keywords': ['synchronization', 'locks', 'mutex', 'atomic']},
            {'id': 17, 'question': 'Explain the working of a Load Balancer.', 'expected_keywords': ['distribution', 'round robin', 'health checks']},
            {'id': 18, 'question': 'What are the benefits of using Docker and Containers?', 'expected_keywords': ['isolation', 'portability', 'efficiency']},
            {'id': 19, 'question': 'How would you secure a REST API?', 'expected_keywords': ['JWT', 'OAuth', 'HTTPS', 'rate limiting']},
            {'id': 20, 'question': 'Explain the CAP theorem in distributed systems.', 'expected_keywords': ['Consistency', 'Availability', 'Partition Tolerance']}
        ],
        'advanced': [
            {'id': 21, 'question': 'Explain the Consensus algorithm (Paxos or Raft) in distributed systems.', 'expected_keywords': ['leader election', 'replication', 'quorum']},
            {'id': 22, 'question': 'How would you design a system that handles 1 million concurrent users?', 'expected_keywords': ['horizontal scaling', 'CDN', 'database sharding']},
            {'id': 23, 'question': 'Discuss the trade-offs between GraphQL and REST.', 'expected_keywords': ['over-fetching', 'strongly typed', 'caching']},
            {'id': 24, 'question': 'How do you optimize performance at the database level for large datasets?', 'expected_keywords': ['partitioning', 'indexing', 'query optimization']},
            {'id': 25, 'question': 'Explain the internals of a modern search engine (indexing, crawling, ranking).', 'expected_keywords': ['inverted index', 'TF-IDF', 'pagerank']},
            {'id': 26, 'question': 'How do you manage schema migrations in a zero-downtime environment?', 'expected_keywords': ['blue-green', 'backward compatibility', 'feature flags']},
            {'id': 27, 'question': 'Design a distributed rate limiter.', 'expected_keywords': ['Token Bucket', 'Sliding Window', 'Redis']},
            {'id': 28, 'question': 'How would you handle eventual consistency in a globally distributed system?', 'expected_keywords': ['CRDTs', 'vector clocks', 'conflict resolution']},
            {'id': 29, 'question': 'Explain the differences between Monolithic, Microservices, and Serverless architectures.', 'expected_keywords': ['cold start', 'operational overhead', 'scalability']},
            {'id': 30, 'question': 'How do you implement distributed tracing and observability?', 'expected_keywords': ['Jaeger', 'Prometheus', 'Span ID', 'Trace ID']}
        ]
    },
    'data_scientist': {
        'role': 'Data Scientist',
        'description': 'Machine Learning and Data Analysis',
        'category': 'job',
        'self_introduction': {
            'question': 'Tell me about your journey in Data Science',
            'guidance': 'Highlight: projects, tools (Python/R), and impact of your analysis',
            'tips': ['Mention your favorite ML algorithm', 'Talk about data cleaning importance', 'Focus on business value']
        },
        'beginner': [
            {'id': 1, 'question': 'What is the difference between supervised and unsupervised learning?', 'expected_keywords': ['labeled data', 'clustering', 'regression', 'classification']},
            {'id': 2, 'question': 'Explain the concept of Overfitting and how to prevent it.', 'expected_keywords': ['regularization', 'cross-validation', 'test set']},
            {'id': 3, 'question': 'What are the main assumptions of Linear Regression?', 'expected_keywords': ['linearity', 'homoscedasticity', 'independence', 'normality']},
            {'id': 4, 'question': 'What is a Confusion Matrix?', 'expected_keywords': ['Precision', 'Recall', 'Accuracy', 'F1-score']},
            {'id': 5, 'question': 'Explain Bias-Variance tradeoff.', 'expected_keywords': ['complexity', 'error', 'underfitting', 'overfitting']},
            {'id': 6, 'question': 'What is p-value in statistics?', 'expected_keywords': ['significance', 'null hypothesis', 'probability']},
            {'id': 7, 'question': 'Explain the difference between L1 and L2 regularization.', 'expected_keywords': ['Lasso', 'Ridge', 'sparsity', 'penalty']},
            {'id': 8, 'question': 'What is Feature Engineering?', 'expected_keywords': ['transformation', 'selection', 'domain knowledge']},
            {'id': 9, 'question': 'Explain K-means clustering.', 'expected_keywords': ['centroids', 'unsupervised', 'distance']},
            {'id': 10, 'question': 'What is the Central Limit Theorem?', 'expected_keywords': ['sampling distribution', 'normal distribution', 'mean']}
        ],
        'intermediate': [
            {'id': 11, 'question': 'How do you handle missing values in a dataset?', 'expected_keywords': ['imputation', 'deletion', 'mean/median/mode']},
            {'id': 12, 'question': 'Explain the working of Random Forest.', 'expected_keywords': ['ensemble', 'decision trees', 'bagging']},
            {'id': 13, 'question': 'What is Gradient Descent and how does it work?', 'expected_keywords': ['learning rate', 'optimization', 'cost function']},
            {'id': 14, 'question': 'Explain the difference between Bagging and Boosting.', 'expected_keywords': ['parallel', 'sequential', 'variance', 'bias']},
            {'id': 15, 'question': 'How do you evaluate a model with highly imbalanced data?', 'expected_keywords': ['SMOTE', 'AUC-ROC', 'Precision-Recall curve']},
            {'id': 16, 'question': 'What is PCA (Principal Component Analysis)?', 'expected_keywords': ['dimensionality reduction', 'variance', 'eigenvectors']},
            {'id': 17, 'question': 'Explain Time Series forecasting.', 'expected_keywords': ['seasonality', 'trend', 'ARIMA', 'Prophet']},
            {'id': 18, 'question': 'What are Hyperparameters and how do you tune them?', 'expected_keywords': ['GridSearch', 'RandomSearch', 'Bayesian']},
            {'id': 19, 'question': 'Explain Cross-Validation.', 'expected_keywords': ['k-fold', 'validation set', 'model robustness']},
            {'id': 20, 'question': 'How do you communicate complex technical findings to non-technical stakeholders?', 'expected_keywords': ['visualization', 'storytelling', 'business metrics']}
        ],
        'advanced': [
            {'id': 21, 'question': 'Explain the architecture of Transformers in NLP.', 'expected_keywords': ['attention mechanism', 'encoder', 'decoder']},
            {'id': 22, 'question': 'How do you deal with the "Curse of Dimensionality"?', 'expected_keywords': ['feature selection', 'regularization', 'manifold learning']},
            {'id': 23, 'question': 'Discuss the Vanishing Gradient problem in Deep Learning.', 'expected_keywords': ['activation functions', 'LSTM', 'ResNet']},
            {'id': 24, 'question': 'How would you deploy a machine learning model into production?', 'expected_keywords': ['Docker', 'CI/CD', 'API', 'monitoring']},
            {'id': 25, 'question': 'Explain A/B testing and how to calculate sample size.', 'expected_keywords': ['power', 'effect size', 'significance level']},
            {'id': 26, 'question': 'Discuss the ethical implications of AI and Bias.', 'expected_keywords': ['fairness', 'transparency', 'explainability']},
            {'id': 27, 'question': 'Explain Reinforcement Learning.', 'expected_keywords': ['agent', 'reward', 'environment', 'policy']},
            {'id': 28, 'question': 'How do you build a recommendation system?', 'expected_keywords': ['collaborative filtering', 'content-based', 'matrix factorization']},
            {'id': 29, 'question': 'Explain Graph Neural Networks.', 'expected_keywords': ['nodes', 'edges', 'embeddings', 'connectivity']},
            {'id': 30, 'question': 'How do you handle cold-start problems in recommendation systems?', 'expected_keywords': ['hybrid systems', 'demographics', 'popularity']}
        ]
    },
    'product_manager': {
        'role': 'Product Manager',
        'description': 'Product Strategy and Lifecycle Management',
        'category': 'job',
        'self_introduction': {
            'question': 'What makes you a great Product Manager?',
            'guidance': 'Focus on: user empathy, data-driven decisions, and cross-functional leadership',
            'tips': ['Mention a product you launched', 'Talk about prioritizing features', 'Focus on vision']
        },
        'beginner': [
            {'id': 1, 'question': 'What is a Minimum Viable Product (MVP)?', 'expected_keywords': ['core features', 'user feedback', 'validation']},
            {'id': 2, 'question': 'How do you prioritize a product backlog?', 'expected_keywords': ['MoSCoW', 'RICE', 'business value']},
            {'id': 3, 'question': 'What are the main stages of the Product Lifecycle?', 'expected_keywords': ['introduction', 'growth', 'maturity', 'decline']},
            {'id': 4, 'question': 'How do you define product success?', 'expected_keywords': ['KPIs', 'metrics', 'user adoption']},
            {'id': 5, 'question': 'What is User Journey Mapping?', 'expected_keywords': ['touchpoints', 'pain points', 'customer experience']},
            {'id': 6, 'question': 'Explain the difference between a Product Roadmap and a Backlog.', 'expected_keywords': ['vision', 'strategy', 'execution', 'tasks']},
            {'id': 7, 'question': 'How do you handle stakeholders with conflicting priorities?', 'expected_keywords': ['negotiation', 'data', 'transparency']},
            {'id': 8, 'question': 'What is Agile methodology?', 'expected_keywords': ['scrum', 'sprint', 'iterative', 'flexibility']},
            {'id': 9, 'question': 'Explain the role of User Research in product development.', 'expected_keywords': ['interviews', 'surveys', 'personas']},
            {'id': 10, 'question': 'What are the basic types of product metrics?', 'expected_keywords': ['acquisition', 'retention', 'revenue', 'engagement']}
        ],
        'intermediate': [
            {'id': 11, 'question': 'How do you decide when to sunset a product or feature?', 'expected_keywords': ['ROI', 'usage data', 'strategic fit']},
            {'id': 12, 'question': 'Explain the GTM (Go-To-Market) strategy.', 'expected_keywords': ['pricing', 'distribution', 'marketing', 'target audience']},
            {'id': 13, 'question': 'How do you conduct a competitive analysis?', 'expected_keywords': ['SWOT', 'benchmarking', 'market trends']},
            {'id': 14, 'question': 'Explain Product-Market Fit.', 'expected_keywords': ['retention', 'word of mouth', 'market demand']},
            {'id': 15, 'question': 'How do you use data to drive product improvements?', 'expected_keywords': ['A/B testing', 'analytics', 'funnel analysis']},
            {'id': 16, 'question': 'Explain the North Star Metric.', 'expected_keywords': ['value proposition', 'long-term growth', 'alignment']},
            {'id': 17, 'question': 'How do you manage a product through technical debt?', 'expected_keywords': ['refactoring', 'trade-offs', 'long-term health']},
            {'id': 18, 'question': 'What is the "First Principles" thinking in product management?', 'expected_keywords': ['assumptions', 'root cause', 'originality']},
            {'id': 19, 'question': 'How do you scale a product from 1k to 1M users?', 'expected_keywords': ['infrastructure', 'automation', 'virality']},
            {'id': 20, 'question': 'Explain the Kano Model for feature prioritization.', 'expected_keywords': ['satisfaction', 'delighters', 'basic needs']}
        ],
        'advanced': [
            {'id': 21, 'question': 'How would you handle a major product pivot?', 'expected_keywords': ['risk assessment', 'communication', 'vision alignment']},
            {'id': 22, 'question': 'Discuss the ethics of "Dark Patterns" in product design.', 'expected_keywords': ['manipulation', 'transparency', 'user trust']},
            {'id': 23, 'question': 'How do you build a product ecosystem or platform?', 'expected_keywords': ['network effects', 'APIs', 'third-party developers']},
            {'id': 24, 'question': 'Explain the differences between B2B and B2C product management.', 'expected_keywords': ['sales cycles', 'user vs buyer', 'customization']},
            {'id': 25, 'question': 'How do you lead a team through a crisis or product failure?', 'expected_keywords': ['ownership', 'learning', 'morale']},
            {'id': 26, 'question': 'Discuss the impact of AI/ML on the future of Product Management.', 'expected_keywords': ['personalization', 'automation', 'predictive analytics']},
            {'id': 27, 'question': 'How do you manage cross-platform product consistency?', 'expected_keywords': ['design systems', 'uniformity', 'platform nuances']},
            {'id': 28, 'question': 'Explain the concept of "Jobs to be Done" (JTBD).', 'expected_keywords': ['motivation', 'progress', 'switching costs']},
            {'id': 29, 'question': 'How do you balance short-term revenue goals with long-term product vision?', 'expected_keywords': ['strategy', 'trade-offs', 'sustainable growth']},
            {'id': 30, 'question': 'Design a global product from scratch. What are the key considerations?', 'expected_keywords': ['localization', 'regulation', 'latency', 'culture']}
        ]
    },
    'sales_executive': {
        'role': 'Sales Executive',
        'description': 'Sales, Negotiation, and Business Development',
        'category': 'job',
        'self_introduction': {
            'question': 'What is your sales philosophy?',
            'guidance': 'Focus on: relationship building, problem solving, and results',
            'tips': ['Mention your biggest win', 'Talk about persistence', 'Focus on customer needs']
        },
        'beginner': [
            {'id': 1, 'question': 'What is the Sales Funnel?', 'expected_keywords': ['prospecting', 'leads', 'conversion', 'closing']},
            {'id': 2, 'question': 'How do you handle common sales objections?', 'expected_keywords': ['listening', 'empathy', 'value proposition']},
            {'id': 3, 'question': 'What is the difference between a lead and a prospect?', 'expected_keywords': ['qualification', 'intent', 'fit']},
            {'id': 4, 'question': 'How do you build rapport with a potential client?', 'expected_keywords': ['listening', 'shared interests', 'professionalism']},
            {'id': 5, 'question': 'Explain the "Elevator Pitch".', 'expected_keywords': ['concise', 'hook', 'value statement']},
            {'id': 6, 'question': 'What is Cold Calling?', 'expected_keywords': ['outreach', 'persistence', 'scripting']},
            {'id': 7, 'question': 'How do you qualify a lead?', 'expected_keywords': ['BANT', 'budget', 'authority', 'need', 'timeline']},
            {'id': 8, 'question': 'What is a CRM and why is it important?', 'expected_keywords': ['tracking', 'relationships', 'data-driven']},
            {'id': 9, 'question': 'Explain the importance of follow-up in sales.', 'expected_keywords': ['persistence', 'touchpoints', 'conversion']},
            {'id': 10, 'question': 'What is Social Selling?', 'expected_keywords': ['LinkedIn', 'content', 'networking']}
        ],
        'intermediate': [
            {'id': 11, 'question': 'How do you manage a long and complex sales cycle?', 'expected_keywords': ['nurturing', 'stakeholders', 'milestones']},
            {'id': 12, 'question': 'Explain Value-Based Selling.', 'expected_keywords': ['ROI', 'outcomes', 'customer success']},
            {'id': 13, 'question': 'How do you negotiate when the client asks for a steep discount?', 'expected_keywords': ['trade-offs', 'value focus', 'anchoring']},
            {'id': 14, 'question': 'What is Solution Selling?', 'expected_keywords': ['diagnosis', 'pain points', 'customization']},
            {'id': 15, 'question': 'How do you research a client before a meeting?', 'expected_keywords': ['annual reports', 'news', 'LinkedIn', 'competitors']},
            {'id': 16, 'question': 'Explain the "Always Be Closing" (ABC) mindset vs modern sales.', 'expected_keywords': ['pressure', 'consultative', 'timing']},
            {'id': 17, 'question': 'How do you handle losing a major deal?', 'expected_keywords': ['analysis', 'resilience', 'learning']},
            {'id': 18, 'question': 'What is Cross-selling and Up-selling?', 'expected_keywords': ['incremental value', 'expansion', 'existing customers']},
            {'id': 19, 'question': 'How do you manage your sales quota and pipeline?', 'expected_keywords': ['forecasting', 'velocity', 'activity metrics']},
            {'id': 20, 'question': 'Explain the SPIN selling technique.', 'expected_keywords': ['Situation', 'Problem', 'Implication', 'Need-payoff']}
        ],
        'advanced': [
            {'id': 21, 'question': 'How would you enter a completely new market for a product?', 'expected_keywords': ['segmentation', 'partnerships', 'localization']},
            {'id': 22, 'question': 'Discuss the role of Emotional Intelligence (EQ) in high-stakes sales.', 'expected_keywords': ['self-awareness', 'negotiation', 'influence']},
            {'id': 23, 'question': 'How do you manage a global sales team with different cultures?', 'expected_keywords': ['adaptability', 'communication', 'KPI alignment']},
            {'id': 24, 'question': 'Explain Enterprise Selling.', 'expected_keywords': ['multi-stakeholder', 'political mapping', 'long-term strategy']},
            {'id': 25, 'question': 'How do you build a sustainable referral network?', 'expected_keywords': ['advocacy', 'quality', 'reciprocity']},
            {'id': 26, 'question': 'Discuss the impact of AI on the Sales profession.', 'expected_keywords': ['automation', 'personalization', 'intelligence']},
            {'id': 27, 'question': 'How do you navigate a "No-Decision" outcome from a prospect?', 'expected_keywords': ['urgency', 'cost of inaction', 're-engagement']},
            {'id': 28, 'question': 'Explain Challenger Selling.', 'expected_keywords': ['teach', 'tailor', 'take control']},
            {'id': 29, 'question': 'How do you design a sales compensation plan?', 'expected_keywords': ['incentives', 'alignment', 'caps/floors']},
            {'id': 30, 'question': 'Describe your method for strategic account management.', 'expected_keywords': ['growth', 'retention', 'executive alignment']}
        ]
    },
    'uiux_designer': {
        'role': 'UI/UX Designer',
        'description': 'User Interface and Experience Design',
        'category': 'job',
        'self_introduction': {
            'question': 'How do you define your design style?',
            'guidance': 'Focus on: user-centricity, problem solving, and aesthetics',
            'tips': ['Mention your favorite design tool', 'Talk about empathy', 'Show passion for detail']
        },
        'beginner': [
            {'id': 1, 'question': 'What is the difference between UI and UX?', 'expected_keywords': ['visual', 'functional', 'interaction', 'feel']},
            {'id': 2, 'question': 'Explain the Design Thinking process.', 'expected_keywords': ['empathize', 'define', 'ideate', 'prototype', 'test']},
            {'id': 3, 'question': 'What are Wireframes?', 'expected_keywords': ['layout', 'low-fidelity', 'structure']},
            {'id': 4, 'question': 'What is Visual Hierarchy?', 'expected_keywords': ['contrast', 'size', 'color', 'spacing']},
            {'id': 5, 'question': 'Explain the importance of Color Theory in design.', 'expected_keywords': ['psychology', 'accessibility', 'branding']},
            {'id': 6, 'question': 'What is Typography and why does it matter?', 'expected_keywords': ['readability', 'legibility', 'hierarchy']},
            {'id': 7, 'question': 'What are Design Systems?', 'expected_keywords': ['components', 'consistency', 'library']},
            {'id': 8, 'question': 'Explain Accessibility (A11y) in web design.', 'expected_keywords': ['contrast', 'screen readers', 'inclusivity']},
            {'id': 9, 'question': 'What is Prototyping?', 'expected_keywords': ['interaction', 'validation', 'flow']},
            {'id': 10, 'question': 'What is the "Mobile-First" approach?', 'expected_keywords': ['constraints', 'scaling', 'prioritization']}
        ],
        'intermediate': [
            {'id': 11, 'question': 'How do you handle negative feedback on your designs?', 'expected_keywords': ['objectivity', 'iteration', 'communication']},
            {'id': 12, 'question': 'Explain the concept of Information Architecture (IA).', 'expected_keywords': ['navigation', 'taxonomy', 'organization']},
            {'id': 13, 'question': 'How do you conduct Usability Testing?', 'expected_keywords': ['tasks', 'observations', 'insights']},
            {'id': 14, 'question': 'Explain Responsive vs Adaptive design.', 'expected_keywords': ['fluid grids', 'breakpoints', 'fixed layouts']},
            {'id': 15, 'question': 'What are Micro-interactions?', 'expected_keywords': ['feedback', 'delight', 'details']},
            {'id': 16, 'question': 'Explain the Gestalt Principles.', 'expected_keywords': ['proximity', 'similarity', 'closure', 'continuity']},
            {'id': 17, 'question': 'How do you design for different platforms (iOS, Android, Web)?', 'expected_keywords': ['guidelines', 'patterns', 'constraints']},
            {'id': 18, 'question': 'What is User-Centered Design (UCD)?', 'expected_keywords': ['empathy', 'requirements', 'feedback']},
            {'id': 19, 'question': 'How do you balance business goals with user needs?', 'expected_keywords': ['trade-offs', 'alignment', 'conversion']},
            {'id': 20, 'question': 'Explain the concept of "Cognitive Load".', 'expected_keywords': ['simplicity', 'mental effort', 'clutter']}
        ],
        'advanced': [
            {'id': 21, 'question': 'How do you design for Voice User Interfaces (VUI)?', 'expected_keywords': ['conversational', 'utterances', 'feedback']},
            {'id': 22, 'question': 'Discuss the future of AR/VR in UX design.', 'expected_keywords': ['spatial', 'immersion', 'interaction']},
            {'id': 23, 'question': 'How do you measure the ROI of UX design?', 'expected_keywords': ['conversion rate', 'support tickets', 'user satisfaction']},
            {'id': 24, 'question': 'Discuss the ethics of "Persuasive Design".', 'expected_keywords': ['addiction', 'transparency', 'autonomy']},
            {'id': 25, 'question': 'How do you build and maintain a cross-functional Design System?', 'expected_keywords': ['governance', 'adoption', 'scalability']},
            {'id': 26, 'question': 'Explain "Emotional Design".', 'expected_keywords': ['visceral', 'behavioral', 'reflective']},
            {'id': 27, 'question': 'How do you design for complex data visualization?', 'expected_keywords': ['clarity', 'exploration', 'interactivity']},
            {'id': 28, 'question': 'Discuss the impact of AI on the design workflow.', 'expected_keywords': ['generative', 'automation', 'personalization']},
            {'id': 29, 'question': 'How do you approach a total brand redesign?', 'expected_keywords': ['identity', 'strategy', 'consistency']},
            {'id': 30, 'question': 'Explain "Service Design" vs Product Design.', 'expected_keywords': ['ecosystem', 'touchpoints', 'end-to-end']}
        ]
    },
    'marketing_manager': {
        'role': 'Marketing Manager',
        'description': 'Marketing Strategy, Branding, and Growth',
        'category': 'job',
        'self_introduction': {
            'question': 'What is your marketing superpower?',
            'guidance': 'Focus on: data analysis, creativity, or brand storytelling',
            'tips': ['Mention a successful campaign', 'Talk about target audience', 'Focus on growth']
        },
        'beginner': [
            {'id': 1, 'question': 'What are the 4 Ps of Marketing?', 'expected_keywords': ['Product', 'Price', 'Place', 'Promotion']},
            {'id': 2, 'question': 'What is Content Marketing?', 'expected_keywords': ['value', 'engagement', 'consistency']},
            {'id': 3, 'question': 'Explain the difference between SEO and SEM.', 'expected_keywords': ['organic', 'paid', 'search engines']},
            {'id': 4, 'question': 'What is a Target Audience?', 'expected_keywords': ['demographics', 'psychographics', 'segmentation']},
            {'id': 5, 'question': 'Explain Brand Identity.', 'expected_keywords': ['logo', 'voice', 'values', 'perception']},
            {'id': 6, 'question': 'What is Social Media Marketing?', 'expected_keywords': ['platforms', 'engagement', 'advertising']},
            {'id': 7, 'question': 'Explain Email Marketing.', 'expected_keywords': ['nurturing', 'segmentation', 'conversion']},
            {'id': 8, 'question': 'What is a Marketing Funnel?', 'expected_keywords': ['awareness', 'consideration', 'conversion']},
            {'id': 9, 'question': 'What is Influencer Marketing?', 'expected_keywords': ['trust', 'reach', 'collaboration']},
            {'id': 10, 'question': 'Explain the importance of Market Research.', 'expected_keywords': ['insights', 'trends', 'competitors']}
        ],
        'intermediate': [
            {'id': 11, 'question': 'How do you measure the success of a marketing campaign?', 'expected_keywords': ['ROI', 'CPA', 'LTV', 'CTR']},
            {'id': 12, 'question': 'Explain Growth Hacking.', 'expected_keywords': ['experimentation', 'virality', 'retention']},
            {'id': 13, 'question': 'How do you manage a marketing budget?', 'expected_keywords': ['allocation', 'tracking', 'optimization']},
            {'id': 14, 'question': 'Explain the concept of "Inbound Marketing".', 'expected_keywords': ['attraction', 'content', 'value-first']},
            {'id': 15, 'question': 'How do you develop a Brand Voice?', 'expected_keywords': ['personality', 'consistency', 'audience alignment']},
            {'id': 16, 'question': 'Explain Marketing Automation.', 'expected_keywords': ['workflows', 'efficiency', 'scalability']},
            {'id': 17, 'question': 'How do you handle a PR crisis?', 'expected_keywords': ['transparency', 'speed', 'communication']},
            {'id': 18, 'question': 'What is Product Positioning?', 'expected_keywords': ['differentiation', 'value proposition', 'competition']},
            {'id': 19, 'question': 'How do you use data for personalized marketing?', 'expected_keywords': ['segmentation', 'behavioral data', 'dynamic content']},
            {'id': 20, 'question': 'Explain the Customer Acquisition Cost (CAC) vs Lifetime Value (LTV).', 'expected_keywords': ['profitability', 'sustainability', 'ratio']}
        ],
        'advanced': [
            {'id': 21, 'question': 'How would you rebrand a legacy company for a younger generation?', 'expected_keywords': ['modernization', 'values', 'channels']},
            {'id': 22, 'question': 'Discuss the role of AI in predictive marketing.', 'expected_keywords': ['algorithms', 'churn prediction', 'personalization']},
            {'id': 23, 'question': 'How do you build a global marketing strategy across different cultures?', 'expected_keywords': ['localization', 'sensitivity', 'consistency']},
            {'id': 24, 'question': 'Discuss the ethics of "Behavioral Targeting".', 'expected_keywords': ['privacy', 'transparency', 'consent']},
            {'id': 25, 'question': 'How do you build a community around a brand?', 'expected_keywords': ['engagement', 'advocacy', 'belonging']},
            {'id': 26, 'question': 'Explain Performance Marketing vs Brand Marketing.', 'expected_keywords': ['short-term', 'long-term', 'measurable']},
            {'id': 27, 'question': 'How do you manage cross-channel attribution?', 'expected_keywords': ['models', 'touchpoints', 'data-driven']},
            {'id': 28, 'question': 'Discuss the impact of "Cookieless" future on digital marketing.', 'expected_keywords': ['first-party data', 'privacy', 'alternative tracking']},
            {'id': 29, 'question': 'How do you design a marketing organization for scale?', 'expected_keywords': ['structure', 'ops', 'specialization']},
            {'id': 30, 'question': 'Explain the concept of "Omnichannel" marketing.', 'expected_keywords': ['seamless', 'integration', 'customer-centric']}
        ]
    },
    'hr_manager': {
        'role': 'HR Manager',
        'description': 'Human Resources, Recruitment, and Employee Relations',
        'category': 'job',
        'self_introduction': {
            'question': 'What is your approach to people management?',
            'guidance': 'Focus on: empathy, compliance, and organizational growth',
            'tips': ['Mention a difficult HR case you solved', 'Talk about company culture', 'Focus on talent']
        },
        'beginner': [
            {'id': 1, 'question': 'What are the main functions of an HR department?', 'expected_keywords': ['recruitment', 'payroll', 'training', 'employee relations']},
            {'id': 2, 'question': 'Explain the Recruitment process.', 'expected_keywords': ['sourcing', 'screening', 'interviewing', 'onboarding']},
            {'id': 3, 'question': 'What is Employee Onboarding?', 'expected_keywords': ['integration', 'culture', 'training', 'tools']},
            {'id': 4, 'question': 'Explain the importance of Performance Reviews.', 'expected_keywords': ['feedback', 'growth', 'alignment', 'goals']},
            {'id': 5, 'question': 'What is Company Culture?', 'expected_keywords': ['values', 'behaviors', 'environment']},
            {'id': 6, 'question': 'How do you handle a basic employee grievance?', 'expected_keywords': ['listening', 'neutrality', 'resolution']},
            {'id': 7, 'question': 'What are Employment Contracts?', 'expected_keywords': ['legal', 'terms', 'obligations']},
            {'id': 8, 'question': 'Explain the importance of Diversity and Inclusion (D&I).', 'expected_keywords': ['equity', 'perspective', 'innovation']},
            {'id': 9, 'question': 'What is Payroll Management?', 'expected_keywords': ['salaries', 'taxes', 'compliance']},
            {'id': 10, 'question': 'Explain the role of Training and Development.', 'expected_keywords': ['upskilling', 'retention', 'productivity']}
        ],
        'intermediate': [
            {'id': 11, 'question': 'How do you handle a difficult termination process?', 'expected_keywords': ['documentation', 'legal compliance', 'professionalism']},
            {'id': 12, 'question': 'Explain Talent Management strategy.', 'expected_keywords': ['succession planning', 'high-potentials', 'retention']},
            {'id': 13, 'question': 'How do you manage organizational change?', 'expected_keywords': ['communication', 'buy-in', 'support']},
            {'id': 14, 'question': 'Explain Employee Engagement and how to measure it.', 'expected_keywords': ['surveys', 'eNPS', 'feedback loops']},
            {'id': 15, 'question': 'How do you conduct a job analysis for a new role?', 'expected_keywords': ['requirements', 'responsibilities', 'competencies']},
            {'id': 16, 'question': 'Explain Compensation and Benefits strategy.', 'expected_keywords': ['market data', 'benchmarking', 'incentives']},
            {'id': 17, 'question': 'How do you mediate a conflict between two senior leaders?', 'expected_keywords': ['neutrality', 'facilitation', 'common goals']},
            {'id': 18, 'question': 'What is Employer Branding?', 'expected_keywords': ['reputation', 'value proposition', 'attraction']},
            {'id': 19, 'question': 'How do you use HR data/analytics for decision making?', 'expected_keywords': ['turnover rate', 'time to hire', 'demographics']},
            {'id': 20, 'question': 'Explain the concept of "Strategic HR".', 'expected_keywords': ['business alignment', 'vision', 'long-term growth']}
        ],
        'advanced': [
            {'id': 21, 'question': 'How would you build a company culture from scratch in a remote-first startup?', 'expected_keywords': ['communication', 'rituals', 'trust']},
            {'id': 22, 'question': 'Discuss the future of work (Remote vs Hybrid vs Office).', 'expected_keywords': ['flexibility', 'productivity', 'connection']},
            {'id': 23, 'question': 'How do you handle a major sexual harassment or ethical investigation?', 'expected_keywords': ['confidentiality', 'impartiality', 'legal action']},
            {'id': 24, 'question': 'Discuss the role of AI in automated recruitment and its bias.', 'expected_keywords': ['fairness', 'transparency', 'human touch']},
            {'id': 25, 'question': 'How do you design a global mobility and relocation policy?', 'expected_keywords': ['tax', 'immigration', 'support']},
            {'id': 26, 'question': 'Explain "Labor Relations" and collective bargaining.', 'expected_keywords': ['unions', 'negotiation', 'contracts']},
            {'id': 27, 'question': 'How do you manage mental health and wellness at an organizational level?', 'expected_keywords': ['support systems', 'culture', 'EAP']},
            {'id': 28, 'question': 'Discuss the "Great Resignation" and its impact on HR strategy.', 'expected_keywords': ['retention', 'flexibility', 'purpose']},
            {'id': 29, 'question': 'How do you design an executive compensation and equity plan?', 'expected_keywords': ['vesting', 'stock options', 'performance metrics']},
            {'id': 30, 'question': 'Explain "Human Capital" as a strategic asset.', 'expected_keywords': ['investment', 'growth', 'competitive advantage']}
        ]
    },
    'f1_student_visa': {
        'role': 'F1 Student Visa',
        'description': 'US Student Visa Interview Preparation',
        'category': 'visa',
        'self_introduction': {
            'question': 'Which university are you going to and why did you choose it?',
            'guidance': 'Focus on: curriculum, research opportunities, and career goals',
            'tips': ['Mention specific professors or labs', 'Talk about the university ranking in your field', 'Show clear academic intent']
        },
        'beginner': [
            {'id': 1, 'question': 'Why do you want to study in the United States?', 'expected_keywords': ['quality of education', 'global exposure', 'advanced research']},
            {'id': 2, 'question': 'How many universities did you apply to and how many admits did you get?', 'expected_keywords': ['admits', 'selection process', 'specific choice']},
            {'id': 3, 'question': 'What is your specialization and why did you choose it?', 'expected_keywords': ['interest', 'future growth', 'skills']},
            {'id': 4, 'question': 'Who is sponsoring your education and what do they do?', 'expected_keywords': ['sponsor', 'income', 'savings', 'affordability']},
            {'id': 5, 'question': 'What are your GRE/TOEFL scores?', 'expected_keywords': ['score', 'aptitude', 'proficiency']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'What are your plans after graduation?', 'expected_keywords': ['return to home country', 'job opportunities', 'contribution']},
            {'id': 7, 'question': 'Why not study this course in your home country?', 'expected_keywords': ['tech gap', 'specialization', 'infrastructure']},
            {'id': 8, 'question': 'How can you prove that you will return to your home country?', 'expected_keywords': ['family ties', 'property', 'employment prospects']},
            {'id': 9, 'question': 'Tell me about the financial status of your sponsor.', 'expected_keywords': ['annual income', 'liquid assets', 'stability']},
            {'id': 10, 'question': 'What if your visa is rejected today?', 'expected_keywords': ['review', 'alternative plans', 'persistence']}
        ],
        'advanced': [
            {'id': 11, 'question': 'Explain your research project in simple terms.', 'expected_keywords': ['methodology', 'impact', 'innovation']},
            {'id': 12, 'question': 'How does this degree align with your long-term 10-year goal?', 'expected_keywords': ['leadership', 'expertise', 'vision']},
            {'id': 13, 'question': 'What is your opinion on the current job market in your field in your home country?', 'expected_keywords': ['growth', 'demand', 'opportunity']},
            {'id': 14, 'question': 'How will you contribute to the diversity on campus?', 'expected_keywords': ['culture', 'collaboration', 'perspective']},
            {'id': 15, 'question': 'Explain the source of any large recent deposits in your sponsor\'s account.', 'expected_keywords': ['transparency', 'evidence', 'legitimacy']}
        ]
    },
    'h1b_work_visa': {
        'role': 'H1B Work Visa',
        'description': 'US Specialty Occupation Work Visa Prep',
        'category': 'visa',
        'self_introduction': {
            'question': 'What is your role and what does your US employer do?',
            'guidance': 'Focus on: specialty occupation, skills, and company value',
            'tips': ['Mention your job title', 'Briefly explain the project', 'Focus on your unique expertise']
        },
        'beginner': [
            {'id': 1, 'question': 'How long have you been working for this company?', 'expected_keywords': ['tenure', 'experience', 'loyalty']},
            {'id': 2, 'question': 'What will be your salary in the United States?', 'expected_keywords': ['prevailing wage', 'compensation', 'benefits']},
            {'id': 3, 'question': 'Where will you be working in the US?', 'expected_keywords': ['location', 'client site', 'headquarters']},
            {'id': 4, 'question': 'What are your core responsibilities in this role?', 'expected_keywords': ['technical skills', 'management', 'delivery']},
            {'id': 5, 'question': 'How did you find this job opportunity?', 'expected_keywords': ['recruitment', 'internal transfer', 'referral']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'Why does this role require a person with your specific background?', 'expected_keywords': ['specialty', 'degree', 'niche skills']},
            {'id': 7, 'question': 'Can you show your LCA (Labor Condition Application)?', 'expected_keywords': ['compliance', 'wage', 'location']},
            {'id': 8, 'question': 'What are your plans if your project ends prematurely?', 'expected_keywords': ['compliance', 'company policy', 'return']},
            {'id': 9, 'question': 'Tell me about the project you will be working on.', 'expected_keywords': ['architecture', 'impact', 'client']},
            {'id': 10, 'question': 'How many employees does your company have in the US?', 'expected_keywords': ['scale', 'operations', 'legitimacy']}
        ],
        'advanced': [
            {'id': 11, 'question': 'Explain the "Specialty Occupation" nature of your work.', 'expected_keywords': ['complexity', 'degree requirement', 'specialized knowledge']},
            {'id': 12, 'question': 'Discuss the right of control your employer has over your work.', 'expected_keywords': ['supervision', 'reviews', 'management']},
            {'id': 13, 'question': 'How does your US salary compare to the industry standard for this location?', 'expected_keywords': ['benchmarking', 'LCA', 'fairness']},
            {'id': 14, 'question': 'What is your opinion on the H1B visa cap and its impact on your industry?', 'expected_keywords': ['policy', 'talent', 'competition']},
            {'id': 15, 'question': 'Explain any previous visa violations or gaps in employment.', 'expected_keywords': ['transparency', 'honesty', 'evidence']}
        ]
    },
    'b1b2_visitor_visa': {
        'role': 'B1/B2 Visitor Visa',
        'description': 'US Tourism and Business Visit Preparation',
        'category': 'visa',
        'self_introduction': {
            'question': 'What is the purpose of your visit to the United States?',
            'guidance': 'Focus on: tourism, family visit, or business meetings',
            'tips': ['Mention specific cities or events', 'Talk about your return plan', 'Be concise']
        },
        'beginner': [
            {'id': 1, 'question': 'Where will you be staying during your visit?', 'expected_keywords': ['hotel', 'friends', 'address']},
            {'id': 2, 'question': 'How long do you plan to stay in the US?', 'expected_keywords': ['duration', 'itinerary', 'dates']},
            {'id': 3, 'question': 'Who is paying for your trip?', 'expected_keywords': ['self-funded', 'company', 'relative']},
            {'id': 4, 'question': 'Have you traveled to any other countries before?', 'expected_keywords': ['travel history', 'compliance', 'passport']},
            {'id': 5, 'question': 'What do you do for a living in your home country?', 'expected_keywords': ['job', 'business', 'income']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'What ties do you have that will ensure your return to your home country?', 'expected_keywords': ['job', 'family', 'property']},
            {'id': 7, 'question': 'Why do you need to spend this much time in the US?', 'expected_keywords': ['itinerary', 'events', 'necessity']},
            {'id': 8, 'question': 'Can you provide a detailed itinerary of your trip?', 'expected_keywords': ['flights', 'bookings', 'plan']},
            {'id': 9, 'question': 'If visiting a relative, what is their legal status in the US?', 'expected_keywords': ['citizenship', 'green card', 'visa']},
            {'id': 10, 'question': 'How much do you expect this trip to cost and do you have the funds?', 'expected_keywords': ['budget', 'savings', 'proof']}
        ],
        'advanced': [
            {'id': 11, 'question': 'Why can\'t these business meetings be conducted virtually?', 'expected_keywords': ['physical presence', 'signing', 'networking']},
            {'id': 12, 'question': 'How do you plan to manage your responsibilities at home while you are away?', 'expected_keywords': ['leave', 'delegation', 'automation']},
            {'id': 13, 'question': 'Discuss any previous US visa denials you may have had.', 'expected_keywords': ['honesty', 'changed circumstances', 'transparency']},
            {'id': 14, 'question': 'How do you plan to handle any medical emergencies during your stay?', 'expected_keywords': ['travel insurance', 'savings', 'support']},
            {'id': 15, 'question': 'Explain the nature of your business and its international operations.', 'expected_keywords': ['trade', 'collaboration', 'expansion']}
        ]
    },
    'financial_analyst': {
        'role': 'Financial Analyst',
        'description': 'Financial Modeling, Analysis, and Reporting',
        'category': 'job',
        'self_introduction': {
            'question': 'Walk me through your experience with financial modeling',
            'guidance': 'Focus on: tools (Excel), accuracy, and decision support',
            'tips': ['Mention a specific model you built', 'Talk about attention to detail', 'Focus on impact']
        },
        'beginner': [
            {'id': 1, 'question': 'What are the three main financial statements?', 'expected_keywords': ['Income Statement', 'Balance Sheet', 'Cash Flow Statement']},
            {'id': 2, 'question': 'Explain the concept of NPV (Net Present Value).', 'expected_keywords': ['discount rate', 'cash flows', 'time value of money']},
            {'id': 3, 'question': 'What is Working Capital?', 'expected_keywords': ['current assets', 'current liabilities', 'liquidity']},
            {'id': 4, 'question': 'Explain the difference between Accrual and Cash accounting.', 'expected_keywords': ['revenue recognition', 'matching principle', 'timing']},
            {'id': 5, 'question': 'What is WACC and why is it important?', 'expected_keywords': ['cost of debt', 'cost of equity', 'weighted average']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'How would you value a company?', 'expected_keywords': ['DCF', 'comparable companies', 'precedent transactions']},
            {'id': 7, 'question': 'Explain Variance Analysis.', 'expected_keywords': ['actual vs budget', 'favorable', 'unfavorable']},
            {'id': 8, 'question': 'What is EBITDA and why do analysts use it?', 'expected_keywords': ['operating performance', 'non-cash expenses', 'comparability']},
            {'id': 9, 'question': 'How do you calculate the Free Cash Flow?', 'expected_keywords': ['operating cash flow', 'capex', 'unlevered']},
            {'id': 10, 'question': 'Explain the impact of a $10 increase in depreciation on the three statements.', 'expected_keywords': ['tax shield', 'net income', 'cash increase']}
        ],
        'advanced': [
            {'id': 11, 'question': 'How do you handle sensitivity analysis in a financial model?', 'expected_keywords': ['variables', 'data tables', 'scenarios']},
            {'id': 12, 'question': 'Discuss the impact of rising interest rates on a company\'s valuation.', 'expected_keywords': ['discount rate', 'debt service', 'equity risk premium']},
            {'id': 13, 'question': 'Explain the M&A accretion/dilution analysis.', 'expected_keywords': ['EPS', 'synergies', 'purchase price']},
            {'id': 14, 'question': 'How do you model a complex debt schedule with tranches?', 'expected_keywords': ['waterfall', 'interest', 'principal repayment']},
            {'id': 15, 'question': 'Discuss the role of financial analysis in strategic business pivoting.', 'expected_keywords': ['opportunity cost', 'breakeven', 'long-term ROI']}
        ]
    },
    'project_manager': {
        'role': 'Project Manager',
        'description': 'Project Planning, Execution, and Monitoring',
        'category': 'job',
        'self_introduction': {
            'question': 'Tell me about a project you led from start to finish',
            'guidance': 'Focus on: scope, timeline, budget, and team leadership',
            'tips': ['Mention specific methodologies used', 'Talk about overcoming a major hurdle', 'Focus on results']
        },
        'beginner': [
            {'id': 1, 'question': 'What is the Critical Path Method (CPM)?', 'expected_keywords': ['dependencies', 'float', 'longest path']},
            {'id': 2, 'question': 'Explain the Project Lifecycle.', 'expected_keywords': ['initiation', 'planning', 'execution', 'closing']},
            {'id': 3, 'question': 'What is a Project Charter?', 'expected_keywords': ['objectives', 'stakeholders', 'authority']},
            {'id': 4, 'question': 'Explain the difference between Agile and Waterfall.', 'expected_keywords': ['iterative', 'sequential', 'flexibility']},
            {'id': 5, 'question': 'What is Scope Creep and how do you prevent it?', 'expected_keywords': ['change control', 'boundaries', 'requirements']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'How do you manage project risks?', 'expected_keywords': ['identification', 'mitigation', 'contingency plan']},
            {'id': 7, 'question': 'Explain the Triple Constraint in project management.', 'expected_keywords': ['time', 'cost', 'scope', 'quality']},
            {'id': 8, 'question': 'How do you handle a team member who is underperforming?', 'expected_keywords': ['communication', 'support', 'accountability']},
            {'id': 9, 'question': 'Explain Earned Value Management (EVM).', 'expected_keywords': ['SPI', 'CPI', 'performance measurement']},
            {'id': 10, 'question': 'How do you manage stakeholder expectations during a project delay?', 'expected_keywords': ['transparency', 'impact analysis', 'revised plan']}
        ],
        'advanced': [
            {'id': 11, 'question': 'How do you manage a portfolio of multiple complex projects?', 'expected_keywords': ['prioritization', 'resource allocation', 'alignment']},
            {'id': 12, 'question': 'Discuss the challenges of managing global, cross-functional teams.', 'expected_keywords': ['time zones', 'culture', 'tools']},
            {'id': 13, 'question': 'How do you implement a PMO (Project Management Office) in an organization?', 'expected_keywords': ['standards', 'governance', 'value']},
            {'id': 14, 'question': 'Explain the concept of "Agile at Scale" (e.g., SAFe).', 'expected_keywords': ['alignment', 'cadence', 'synchronization']},
            {'id': 15, 'question': 'How do you drive project success in a highly uncertain or volatile environment?', 'expected_keywords': ['adaptability', 'lean', 'fast feedback']}
        ]
    },
    'digital_marketing': {
        'role': 'Digital Marketing Specialist',
        'description': 'SEO, PPC, and Social Media Strategy',
        'category': 'job',
        'self_introduction': {
            'question': 'How do you stay updated with digital marketing trends?',
            'guidance': 'Mention: blogs, courses, and experimentation',
            'tips': ['Mention specific algorithms', 'Talk about data-driven decisions', 'Focus on multi-channel']
        },
        'beginner': [
            {'id': 1, 'question': 'What is the difference between SEO and SEM?', 'expected_keywords': ['organic', 'paid', 'search engine']},
            {'id': 2, 'question': 'What are the main types of social media advertising?', 'expected_keywords': ['CPC', 'CPM', 'sponsored content']},
            {'id': 3, 'question': 'Explain the importance of Keywords in SEO.', 'expected_keywords': ['intent', 'relevance', 'volume']},
            {'id': 4, 'question': 'What is an Email Marketing campaign?', 'expected_keywords': ['segmentation', 'open rate', 'conversion']},
            {'id': 5, 'question': 'Explain the concept of "Call to Action" (CTA).', 'expected_keywords': ['conversion', 'button', 'directive']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'How do you optimize a PPC campaign?', 'expected_keywords': ['A/B testing', 'quality score', 'negative keywords']},
            {'id': 7, 'question': 'Explain the role of Content Marketing in a digital strategy.', 'expected_keywords': ['value', 'nurturing', 'inbound']},
            {'id': 8, 'question': 'How do you track and analyze website traffic?', 'expected_keywords': ['Google Analytics', 'conversions', 'bounce rate']},
            {'id': 9, 'question': 'Explain the concept of "Lookalike Audiences".', 'expected_keywords': ['targeting', 'profiling', 'reach']},
            {'id': 10, 'question': 'How do you handle a negative social media comment?', 'expected_keywords': ['professionalism', 'speed', 'resolution']}
        ],
        'advanced': [
            {'id': 11, 'question': 'Discuss the impact of "First-Party Data" in a cookieless world.', 'expected_keywords': ['privacy', 'strategy', 'ownership']},
            {'id': 12, 'question': 'How do you design a global digital marketing ecosystem?', 'expected_keywords': ['localization', 'consistency', 'platforms']},
            {'id': 13, 'question': 'Explain the use of AI and Automation in marketing.', 'expected_keywords': ['predictive analytics', 'chatbots', 'efficiency']},
            {'id': 14, 'question': 'How do you measure the Attribution across multiple channels?', 'expected_keywords': ['models', 'touchpoints', 'data-driven']},
            {'id': 15, 'question': 'Discuss the ethics of data-driven targeting.', 'expected_keywords': ['transparency', 'consent', 'trust']}
        ]
    },
    'cybersecurity_analyst': {
        'role': 'Cybersecurity Analyst',
        'description': 'Network Security and Threat Intelligence',
        'category': 'job',
        'self_introduction': {
            'question': 'What sparked your interest in Cybersecurity?',
            'guidance': 'Focus on: problem solving, ethics, and protection',
            'tips': ['Mention a specific security project', 'Talk about staying updated', 'Focus on vigilance']
        },
        'beginner': [
            {'id': 1, 'question': 'What is the CIA Triad?', 'expected_keywords': ['Confidentiality', 'Integrity', 'Availability']},
            {'id': 2, 'question': 'Explain the difference between a Virus and a Worm.', 'expected_keywords': ['replication', 'host file', 'network']},
            {'id': 3, 'question': 'What is Phishing?', 'expected_keywords': ['social engineering', 'email', 'deception']},
            {'id': 4, 'question': 'Explain the importance of strong passwords.', 'expected_keywords': ['complexity', 'entropy', 'security']},
            {'id': 5, 'question': 'What is a Firewall?', 'expected_keywords': ['traffic filtering', 'network security', 'rules']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'How do you respond to a security incident?', 'expected_keywords': ['containment', 'analysis', 'recovery']},
            {'id': 7, 'question': 'Explain the concept of "Zero Trust" architecture.', 'expected_keywords': ['verify always', 'least privilege', 'security']},
            {'id': 8, 'question': 'What is Encryption and how does it work?', 'expected_keywords': ['algorithms', 'keys', 'plaintext', 'ciphertext']},
            {'id': 9, 'question': 'How do you perform a Vulnerability Assessment?', 'expected_keywords': ['scanning', 'reporting', 'prioritization']},
            {'id': 10, 'question': 'Explain the difference between Symmetric and Asymmetric encryption.', 'expected_keywords': ['keys', 'efficiency', 'security']}
        ],
        'advanced': [
            {'id': 11, 'question': 'Discuss the role of Threat Intelligence in proactive defense.', 'expected_keywords': ['feeds', 'analysis', 'prevention']},
            {'id': 12, 'question': 'How do you secure a cloud-native environment?', 'expected_keywords': ['IAM', 'logging', 'compliance']},
            {'id': 13, 'question': 'Explain the "Kill Chain" in a cyber attack.', 'expected_keywords': ['reconnaissance', 'exploitation', 'actions']},
            {'id': 14, 'question': 'How do you manage security compliance in a global organization?', 'expected_keywords': ['GDPR', 'SOC2', 'governance']},
            {'id': 15, 'question': 'Discuss the impact of Quantum Computing on current encryption.', 'expected_keywords': ['post-quantum', 'algorithms', 'security']}
        ]
    },
    'customer_service': {
        'role': 'Customer Service Representative',
        'description': 'Customer Support, Conflict Resolution, and Communication',
        'category': 'job',
        'self_introduction': {
            'question': 'How do you define excellent customer service?',
            'guidance': 'Focus on: empathy, speed, and accuracy',
            'tips': ['Mention a time you exceeded expectations', 'Talk about active listening', 'Focus on resolution']
        },
        'beginner': [
            {'id': 1, 'question': 'What are the key qualities of a customer service representative?', 'expected_keywords': ['patience', 'empathy', 'communication']},
            {'id': 2, 'question': 'How do you handle an angry customer?', 'expected_keywords': ['listen', 'stay calm', 'apologize', 'resolve']},
            {'id': 3, 'question': 'What is the importance of product knowledge in support?', 'expected_keywords': ['accuracy', 'confidence', 'trust']},
            {'id': 4, 'question': 'Explain the concept of "Active Listening".', 'expected_keywords': ['understanding', 'feedback', 'focus']},
            {'id': 5, 'question': 'How do you handle multiple customer inquiries at once?', 'expected_keywords': ['prioritization', 'speed', 'quality']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'How do you turn a negative customer experience into a positive one?', 'expected_keywords': ['recovery', 'surprise', 'delight']},
            {'id': 7, 'question': 'Explain the role of "Customer Feedback" in business growth.', 'expected_keywords': ['improvement', 'retention', 'insights']},
            {'id': 8, 'question': 'How do you handle a situation where you don\'t know the answer?', 'expected_keywords': ['honesty', 'resourcefulness', 'follow-up']},
            {'id': 9, 'question': 'Explain "Customer Retention" and its importance.', 'expected_keywords': ['loyalty', 'LTV', 'churn reduction']},
            {'id': 10, 'question': 'How do you use CRM tools to improve support?', 'expected_keywords': ['history', 'personalization', 'efficiency']}
        ],
        'advanced': [
            {'id': 11, 'question': 'Discuss the impact of "Self-Service" portals on customer support.', 'expected_keywords': ['efficiency', 'CX', 'automation']},
            {'id': 12, 'question': 'How do you design a customer support strategy for a global product?', 'expected_keywords': ['24/7', 'localization', 'consistency']},
            {'id': 13, 'question': 'Explain "Omnichannel Support" vs "Multichannel Support".', 'expected_keywords': ['seamless', 'integration', 'customer journey']},
            {'id': 14, 'question': 'How do you manage support metrics like CSAT, NPS, and CES?', 'expected_keywords': ['satisfaction', 'loyalty', 'effort']},
            {'id': 15, 'question': 'Discuss the role of AI and Chatbots in the future of Customer Service.', 'expected_keywords': ['automation', 'personalization', 'human escalation']}
        ]
    },
    'business_analyst': {
        'role': 'Business Analyst',
        'description': 'Business Requirements, Process Modeling, and Analysis',
        'category': 'job',
        'self_introduction': {
            'question': 'How do you bridge the gap between business and technology?',
            'guidance': 'Focus on: translation, requirements, and value',
            'tips': ['Mention a specific business problem you solved', 'Talk about stakeholder management', 'Focus on ROI']
        },
        'beginner': [
            {'id': 1, 'question': 'What are the main responsibilities of a Business Analyst?', 'expected_keywords': ['requirements', 'analysis', 'communication']},
            {'id': 2, 'question': 'Explain the difference between a functional and non-functional requirement.', 'expected_keywords': ['behavior', 'performance', 'security']},
            {'id': 3, 'question': 'What is a Use Case?', 'expected_keywords': ['interaction', 'actor', 'goal']},
            {'id': 4, 'question': 'Explain the concept of "Gap Analysis".', 'expected_keywords': ['current state', 'future state', 'bridge']},
            {'id': 5, 'question': 'What is SWOT analysis?', 'expected_keywords': ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']}
        ],
        'intermediate': [
            {'id': 6, 'question': 'How do you conduct a requirements gathering workshop?', 'expected_keywords': ['elicitation', 'facilitation', 'prioritization']},
            {'id': 7, 'question': 'Explain Business Process Modeling (BPMN).', 'expected_keywords': ['workflows', 'optimization', 'visualize']},
            {'id': 8, 'question': 'How do you handle a situation where stakeholders disagree on requirements?', 'expected_keywords': ['negotiation', 'data', 'alignment']},
            {'id': 9, 'question': 'Explain the concept of "User Stories" in Agile.', 'expected_keywords': ['persona', 'need', 'value', 'acceptance criteria']},
            {'id': 10, 'question': 'How do you ensure the technical solution matches the business requirements?', 'expected_keywords': ['validation', 'traceability matrix', 'UAT']}
        ],
        'advanced': [
            {'id': 11, 'question': 'How do you manage change requests in a mid-project phase?', 'expected_keywords': ['impact analysis', 'board', 'prioritization']},
            {'id': 12, 'question': 'Discuss the role of "Data-Driven" decision making in business analysis.', 'expected_keywords': ['analytics', 'metrics', 'evidence']},
            {'id': 13, 'question': 'How do you design a business case for a multi-million dollar investment?', 'expected_keywords': ['ROI', 'NPV', 'risk assessment']},
            {'id': 14, 'question': 'Explain the concept of "Digital Transformation" from a BA perspective.', 'expected_keywords': ['modernization', 'value stream', 'efficiency']},
            {'id': 15, 'question': 'How do you lead a team of junior analysts through a complex project?', 'expected_keywords': ['mentorship', 'standards', 'oversight']}
        ]
    }
}


def get_all_available_roles():
    """Get list of all roles in database"""
    roles = []
    for key, data in INTERVIEW_QUESTIONS.items():
        roles.append({
            'id': key,
            'key': key,
            'name': data['role'],
            'title': data['role'],
            'description': data['description'],
            'category': data['category']
        })
    return roles


def get_role_questions(role_key):
    """Get questions for a specific role"""
    return INTERVIEW_QUESTIONS.get(role_key)


def get_self_introduction_question(role):
    """Get the self-introduction question for a role"""
    data = get_role_questions(role)
    if data and 'self_introduction' in data:
        intro = data['self_introduction']
        return {
            'id': 0,
            'question': intro['question'],
            'guidance': intro.get('guidance', ''),
            'tips': intro.get('tips', []),
            'is_intro': True
        }
    return None


def generate_dynamic_questions(role, level):
    """Generate dynamic questions using Gemini API if needed"""
    try:
        # For now, we use our rich local database
        # This function can be expanded to call Gemini for truly unique questions
        return None
    except Exception as e:
        print(f"Error generating dynamic questions: {e}")
        return None


def get_beginner_questions(role, count=5):
    """Get beginner level questions for a role"""
    data = get_role_questions(role)
    if data:
        questions = data.get('beginner', [])
        # Include self-introduction as the first question if available
        final_questions = []
        if 'self_introduction' in data:
            intro = data['self_introduction']
            final_questions.append({'id': 0, 'question': intro['question'], 'is_intro': True})
        
        # Add the rest of the questions
        shuffled = questions.copy()
        random.shuffle(shuffled)
        final_questions.extend(shuffled)
        return final_questions[:count]
    return []


def get_intermediate_questions(role, count=5):
    """Get intermediate level questions for a role"""
    data = get_role_questions(role)
    if data:
        questions = data.get('intermediate', [])
        shuffled = questions.copy()
        random.shuffle(shuffled)
        return shuffled[:count]
    return []

def get_advanced_questions(role, count=5):
    """Get advanced level questions for a role"""
    data = get_role_questions(role)
    if data:
        questions = data.get('advanced', [])
        shuffled = questions.copy()
        random.shuffle(shuffled)
        return shuffled[:count]
    return []
