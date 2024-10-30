import re

# extract date from uuid
timestamp = "IPRL+w026bb9b+2023-04-20-23h-28m-09s"
regex = r'\w+\+\w+\+(\d+-\d+-\d+-\w+h-\w+m-\w+s)$'
date_str = re.findall(regex, timestamp)[0]
print(date_str)

# match mp4 or txt files
file1 = "recordings/episode_01.txt"
file2 = "metadata/data.mp4"
regex = r'(.*mp4)|(.*txt)'
#print(re.findall(regex, file1))
#print(re.findall(regex, file2))

# match one word or another
# hint: MSDN regular expression
# hint: regex101.com
text1 = "Take out the marker from the silver pot".lower()
text2 = "Remove the marker from the pot and put it on the table".lower()
text3 = "Put the marker in the mug"
regex = r"(take out|remove).*(cup|mug|pot)"
print(re.findall(regex, text1))
print(re.findall(regex, text2))
