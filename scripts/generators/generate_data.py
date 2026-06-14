import json, os

def generate_synthetic_data(num_samples=1000, output_path="data/synthetic"):
    os.makedirs(output_path, exist_ok=True)
    templates = [
        {"role": "user", "templates": ["Explain {topic}", "How does {topic} work?", "What is {topic}?"]},
    ]
    topics = ["machine learning", "transformers", "attention mechanism", "gradient descent",
              "neural networks", "backpropagation", "loss function", "optimizer"]
    samples = []
    for i in range(num_samples):
        topic = topics[i % len(topics)]
        samples.append({
            "messages": [
                {"role": "user", "content": f"Explain {topic} in simple terms."},
                {"role": "assistant", "content": f"{topic} is a fundamental concept in machine learning..."}
            ]
        })
    with open(os.path.join(output_path, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Generated {num_samples} synthetic samples")

if __name__ == "__main__":
    generate_synthetic_data()
