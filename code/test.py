from utils import *
from common_instruct import *

content_type = [
    "Body Shaping",
    "Weight Loss",
    "Women's Health",
    "Men's Health",
    "Health Supplement",
    "Height Growth",
]

print(f"content_type={content_type}")
# gpt4o
compute_accuracy_by_filename("../data/image_test-251119215944-gpt4o-baseline-content_type.json", content_type)
print("X"*200)
compute_accuracy_by_filename("../data/image_test-decompose-251120093951-gpt-41-0414-global-without.json", content_type)
print("X"*200)
compute_accuracy_by_filename("../data/image_test-decompose-251120084333-qwen3-235b-a22b-instruct-2507-without.json", content_type)
print("X"*200)
compute_accuracy_by_filename("../data/image_test-decompose-251120095733-deepseek-r1-without.json", content_type)
print("X"*200)

# claude37
compute_accuracy_by_filename("../data/image_test-251121084506-claude-baseline-content_type.json", content_type)
print("X"*200)
compute_accuracy_by_filename("../data/image_test-decompose-251121005155-gpt-41-0414-global-without.json", content_type)
print("X"*200)
compute_accuracy_by_filename("../data/image_test-decompose-251121011656-qwen3-235b-a22b-instruct-2507-without.json", content_type)
print("X"*200)
compute_accuracy_by_filename("../data/image_test-decompose-251121013559-deepseek-r1-without.json", content_type)
print("X"*200)