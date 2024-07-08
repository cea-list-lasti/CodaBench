with open("test_humaneval.txt", "r") as f:
    prompt = f.readlines()
    prompt = "".join(prompt)
    
print(prompt)