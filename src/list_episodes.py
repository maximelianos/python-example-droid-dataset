import json

def uuid_to_date(uuid: str):
    # uuid looks like "IRIS+ef107c48+2023-03-02-15h-14m-31s"

    regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+-\w+-\w+)$'
    date_str = re.findall(regex, uuid)[0]
    return date_str

# MV existing episodes [date, uuid, path]
with open("data/existing_episodes.json") as f:
    existing_episodes = json.load(f)
date_to_path = {uuid_to_date(episode[1]): episode[2] for episode in existing_episodes}


