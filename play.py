import json

data = {"school": 1543}
print(json.dumps(data))

# file line reading
with open("tmp.txt", "w") as f:
    print("str1", file=f)
    print("str2", file=f)

with open("tmp.txt", "r") as f:
    print([line.strip() for line in f.readlines()])
