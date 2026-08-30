from agent import agent


print("🤖 AI Research Agent")
print("--------------------")

goal = input("What do you want me to do? ")

result = agent.run(goal)

print("\nAgent:")
print(result)
