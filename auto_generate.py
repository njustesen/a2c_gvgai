from level_generator import ParamGenerator

game_sizes = {
    "zelda": [13, 9],
    "boulderdash": [26, 13],
    "frogs": [28, 11],
    "solarfox": [10, 11]
}
game = "boulderdash"
for game in game_sizes:
    for d in [0, 2.5, 7.5]:
        gen = ParamGenerator("./data/test-levels/{}/{}/".format(game, d), game, game_sizes[game][0], game_sizes[game][1])
        for i in range(1):
            level = gen.generate([(d/10.0)], difficulty=True)
            #print(level)
print("Done")