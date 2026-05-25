from utils.indexer import load_indexes
from utils.retriever import perform_hybrid_search

print("Loading Database...")
v_idx, b_idx, c_map, _ = load_indexes("offline_knowledge_base") 

query = "What is the maximum payload and maximum reach for the KR 210 R3100-2 robot?"
print(f"\n🔍 Searching for: '{query}'")

# This is the exact raw output that Mistral will receive
results = perform_hybrid_search(query, v_idx, b_idx, c_map, top_k=3)

for rank, res in enumerate(results):
    print(f"\n--- Rank {rank+1} ---")
    print(f"Source: {res['metadata'].get('source_file', 'Unknown')} | Page: {res['metadata'].get('page_num', 'Unknown')}")
    # Print the first 200 characters to verify it found the right page
    print(f"Preview: {res['text'][:200]}...\n")