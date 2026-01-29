from fastgram import gram
import time

index_dir = "indices/reasoning"
eos_token_id = 151643
vocab_size = 151643
version = 4
token_dtype = "u32"

print(f"Loading index from {index_dir}...")
start = time.time()
# The wrapper class is actually 'gram' which points to GramEngine
# We must use 'index_dir' and 'token_dtype'
engine = gram(
    index_dir=index_dir,
    eos_token_id=eos_token_id,
    vocab_size=vocab_size,
    version=version,
    token_dtype=token_dtype
)
print(f"Index loaded in {time.time() - start:.2f}s")

# Test query for "The" (token 1302)
prompt = [1302]

print(f"Querying NTD for prompt ids: {prompt}")
start = time.time()
result = engine.ntd(prompt, max_support=1000)
end = time.time()

print(f"NTD Result (Top 5):")
items = list(result['result_by_token_id'].items())
items.sort(key=lambda x: x[1]['cont_cnt'], reverse=True)
for tok_id, res in items[:5]:
    print(f"  Token {tok_id}: Count {res['cont_cnt']}, Prob {res['prob']:.4f}")

print(f"Total prompt count: {result['prompt_cnt']}")
print(f"Query took {end - start:.4f}s")
