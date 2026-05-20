from memory import embed, cosine_similarity, load_memory

# Simulate what the agent would do
user_goal = "Find information about the Roman emperor Hadrian — when he was born, when he ruled, and what he's known for."

goal_embedding = embed(user_goal)
lessons = load_memory()

print(f"User goal: {user_goal[:80]}...\n")
print(f"Threshold: 0.35\n")

for lesson in lessons:
    sim = cosine_similarity(goal_embedding, lesson["embedding"])
    status = "RETRIEVED" if sim >= 0.35 else "BELOW THRESHOLD"
    print(f"  [{status}] sim={sim:.3f}")
    print(f"  Lesson: {lesson['lesson'][:120]}...")
    print()