import collections
from tqdm import tqdm
import datasets


def normalize_string(text):
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def word_ngrams(text, n):
    """Generate word-level n-grams from text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

# def load_questions(name, split, question_field):
#     """Load questions from a dataset."""
#     all_questions = []
#     dataset = datasets.load_dataset(name, trust_remote_code=True)[split]
#     for example in tqdm(dataset, desc=f"Loading {name}"):
#         all_questions.append(example[question_field])
#     return all_questions

def build_ngram_lookup(documents, ngram_size=13):
    """Build ngram lookup for documents."""
    print(f"Building {ngram_size}-gram lookup...")
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)
    
    return lookup


def find_contaminated_questions(test_lookup, train_lookup):
    """Find overlapping documents based on ngram matches."""
    contaminated_ids = set()
    matched_ngrams = []  # For debugging
    
    for ngram, test_doc_ids in tqdm(test_lookup.items(), desc="Checking overlaps"):
        if ngram in train_lookup:
            contaminated_ids.update(test_doc_ids)
            matched_ngrams.append(ngram)
    
    # Print some example matches for inspection
    if matched_ngrams:
        print("\nExample matching n-grams:")
        for ngram in matched_ngrams[:5]:  # Show first 5 matches
            print(f"  - {ngram}")
    
    return contaminated_ids



def upload_to_huggingface(selected_examples, repo_id="your-username/dataset-name", source="Omni-MATH"):
    # Format the data for upload
    formatted_data = []
    for ex in tqdm(selected_examples, desc="Formatting examples"):
        try:
            if "domain" in ex or "difficulty" in ex:
                formatted_example = {
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                    "domain": ex["domain"][0],
                    "difficulty": ex["difficulty"],
                    "subdomain": ex["domain"][0].split(" -> ")[2],
                    "source": source
                }
            else:
                formatted_example = {
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                    "source": source,
                    "messages": ex["messages"]
                }
            formatted_data.append(formatted_example)
        except Exception as e:
            print(f"Error formatting example: {e}")
            continue

    # Create the dataset
    dataset = datasets.Dataset.from_list(formatted_data)
    
    # Print dataset info
    print("\nDataset Statistics:")
    print(f"Total examples: {len(dataset)}")
    print("\nFeatures:", dataset.features)
    
    # Push to hub
    try:
        dataset.push_to_hub(repo_id)
        print(f"\nSuccessfully uploaded dataset to {repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        return None

    return dataset