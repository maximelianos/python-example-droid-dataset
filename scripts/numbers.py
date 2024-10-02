import json

valid = []
for i in range(150):
    s = input("Is episode valid? " + str(i)).strip()
    if s:
        valid.append(i)

    with open("data/valid_episodes.json", "w") as f:
        json.dump(valid, f, indent=4, ensure_ascii=False)
