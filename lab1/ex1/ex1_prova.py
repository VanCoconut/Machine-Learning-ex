import math as m
import string as s


class Player:
    scores = list(())

    def __init__(self, name=None, country=None, scores=list(())):
        self.name = name
        self.country = country
        self.scores = scores


f1 = open("scores.txt", "r")
f2 = open("pretty_scores.txt", "w+")
lines = f1.readlines()
print(lines)

for line in lines:
    line = line.split(" ")
    a = Player(line[0] + " " + line[1], line[2], list(()))
    l = list(())
    for i in range(5):
        l.append(line[3 + i])
    l.remove(min(l))
    l.remove(max(l))
    a.scores = l
    f2.write(a.name + " " + a.country + " " + round(sum(map(float, a.scores)), 2).__str__() + "\n")

f1.close()
f2.close()
f2 = open("pretty_scores.txt", "r")
f1 = open("ranking.txt", "w+")
lines = f2.readlines()
topScore = 0
topPlayer = Player()
topCountry = {"country": "", "scores": 0}
for line in lines:
    line = line.split(" ")
    a = Player(line[0] + " " + line[1], line[2], list((line[3])))
    l = list(()).append(a.scores)
    # if max(a.scores) > topScore:
    #     topScore = max(a.scores)
    #     topPlayer = Player(a.name, None, a.scores[0])
    # if topCountry["country"] == a.country :
    #     topCountry["country"] = a.country
    #     topCountry["scores"] += a.scores
    # f1.write(a.name + " " + a.country + " - Score: " + round(sum(map(float, a.scores)), 2).__str__() + "\n")
f1.close()
f2.close()
