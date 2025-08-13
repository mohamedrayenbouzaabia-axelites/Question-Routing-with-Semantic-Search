import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_seeding():
    """Test seeding new classes"""
    seed_data = {
        "classes": [
            {
                "mode": "new_class",
                "class_name": "Technical Support",
                "class_description": "Handle technical issues, bugs, and troubleshooting requests",
                "example_questions": [
                    "My app is crashing when I try to login",
                    "How do I reset my password?",
                    "The website won't load properly"
                ],
                "pool_id": "customer_support"
            },
            {
                "mode": "new_class",
                "class_name": "Billing Inquiries", 
                "class_description": "Handle billing, payment, and subscription related questions",
                "example_questions": [
                    "I was charged twice this month",
                    "How do I cancel my subscription?",
                    "Can I get a refund for last month?"
                ],
                "pool_id": "customer_support"
            },
            {
                "mode": "new_class",
                "class_name": "Product Information",
                "class_description": "Provide information about product features and capabilities",
                "example_questions": [
                    "What features are included in the premium plan?",
                    "Does your software support integrations?",
                    "What are the system requirements?"
                ],
                "pool_id": "customer_support"
            }
        ]
    }
    
    print("Testing seeding...")
    response = requests.post(f"{BASE_URL}/seed", json=seed_data, headers=HEADERS)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def test_queries():
    """Test different query scenarios"""
    queries = [
        {
            "question": "I can't log into my account",
            "pool_id": "customer_support"
        },
        {
            "question": "I need help with my monthly bill",
            "pool_id": "customer_support"
        },
        {
            "question": "What integrations do you support?",
            "pool_id": "customer_support"
        },
        {
            "question": "Can you help me with something completely unrelated?",
            "pool_id": "customer_support"
        }
    ]
    
    print("\nTesting queries...")
    for query in queries:
        print(f"\nQuery: {query['question']}")
        response = requests.post(f"{BASE_URL}/query", json=query, headers=HEADERS)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Search method: {result['search_method']}")
            print("Results:")
            for res in result['results']:
                print(f"  - {res['class_name']}: {res['probability']:.3f}")

def test_batch_seeding():
    """Test batch seeding with mixed modes"""
    # First get some existing class IDs
    pool_response = requests.get(f"{BASE_URL}/pools/customer_support/classes")
    if pool_response.status_code == 200:
        classes = pool_response.json()['classes']
        if classes:
            existing_class_id = classes[0]['class_id']
            
            batch_data = {
                "classes": [
                    {
                        "mode": "existing_class",
                        "class_id": existing_class_id,
                        "example_questions": [
                            "Another technical question example",
                            "One more troubleshooting example"
                        ],
                        "pool_id": "customer_support"
                    },
                    {
                        "mode": "new_class",
                        "class_name": "Sales Inquiries",
                        "class_description": "Handle sales questions and demo requests",
                        "example_questions": [
                            "I'd like to schedule a demo",
                            "What's your pricing model?",
                            "Can I speak with a sales representative?"
                        ],
                        "pool_id": "customer_support"
                    }
                ]
            }
            
            print("\nTesting batch seeding...")
            response = requests.post(f"{BASE_URL}/seed", json=batch_data, headers=HEADERS)
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    # Test the API
    print("Starting API tests...")
    
    # Test health check
    health = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {health.status_code} - {health.json()}")
    
    # Test seeding
    seed_result = test_seeding()
    
    # Test queries
    test_queries()
    
    # Test batch seeding
    test_batch_seeding()
    
    print("\nTests completed!")