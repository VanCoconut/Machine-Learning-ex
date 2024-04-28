class Player:

    def __init__(self, name=None, country=None, scores=None):
        self.name = name
        self.country = country
        self.scores = scores

    def to_string_top3(self):
        return f'{self.name}, Score: {self.scores}'


f1 = open("scores.txt", "r")
lines = f1.readlines()
countries = {}
players = list(())
for line in lines:
    line = line.split(" ")
    a = Player(line[0] + " " + line[1], line[2], list(()))
    l = list(())
    for i in range(5):
        l.append(line[3 + i])
    l.remove(min(l))
    l.remove(max(l))
    a.scores = l
    players.append(a)
for player in players:
    player.scores = round(sum(map(float, player.scores)), 2)
    if not countries.__contains__(player.country):
        countries[player.country] = player.scores
    else:
        countries[player.country] += player.scores
# print(countries)
top3_rank = list(())
for i in range(3):
    top_score = max(players, key=lambda x: x.scores)
    top3_rank.append(top_score)
    players.remove(top_score)
top3_rank.sort(key=lambda x: x.scores, reverse=True)

print("Final ranking:")
for i in range(3):
    print(str(i + 1) + ":", end=" ")
    print(top3_rank[i].to_string_top3())

top_country = max(countries, key=countries.get)
print("\nBestCountry:\n" + top_country + " - Total score: " + countries[top_country].__str__())
