class Person:
    def __init__(self, name, city, date):
        self.name = name
        self.city = city
        self.date = date


f = open('people.txt', 'r')
lines = f.readlines()
people = list(())
cities = {}
months = {}
for line in lines:
    line = line.split(" ")
    person = Person(line[0] + " " + line[1], line[2], line[3])
    people.append(person)
    if not cities.__contains__(line[2]):
        cities[line[2]] = 1
    else:
        cities[line[2]] += 1
    month = line[3].split("/")[1]
    if not months.__contains__(month):
        months[month] = 1
    else:
        months[month] += 1
avg = round(len(people) / len(cities),2)

print("Births for each city: " + str(cities))
print("Births for each month: " + str(months))
print("Average number of births per city: " + str(avg))
