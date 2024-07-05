from pathlib import Path
import subprocess

data = Path("data/droid_raw/1.0.1/success")
episodes = []
for date in data.iterdir():
    for episode in date.iterdir():
        episodes.append(episode)

print(len(episodes))
#episodes = episodes[:10]
for episode in episodes:
    print("episode:", episode)
    plot_path = Path("plot") / (episode.parent.name + "_" + episode.name + ".jpg")
    print("plot path:", plot_path)

    command = ["src/raw.py", "--scene", str(episode), "--plot", plot_path]
    print(f'Running: "{" ".join(map(str, command))}"')
    p: subprocess.CompletedProcess = subprocess.run(command)
