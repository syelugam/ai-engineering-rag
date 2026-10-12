
documents = [
    {
        "title": "Deployment Guide",
        "text": "Production services run on AWS ECS Fargate behind an Application Load Balancer.",
    },
    {
        "title": "Monitoring Guide",
        "text": "Services should publish structured logs and metrics, and alert on elevated error rates.",
    },
    {
        "title": "Security Guide",
        "text": "Applications should use least-privilege IAM roles and keep credentials out of source code.",
    },
]


def retrieve_documents(query: str) -> list[dict]:
    terms = set(query.lower().split())
    scored = []

    for document in documents:
        text = (document["title"] + " " + document["text"]).lower()
        score = sum(1 for term in terms if term in text.split())

        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:2]]


question = "How should AWS services be deployed?"
results = retrieve_documents(question)

print("Question:", question)
print("\nRetrieved documents:")

for document in results:
    print("-", document["title"], ":", document["text"])
