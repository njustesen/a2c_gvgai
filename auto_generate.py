from level_generator import ParamGenerator

game_sizes = {
    "aliens": [30, 11],
    "zelda": [13, 9],
    "boulderdash": [26, 13],
    "frogs": [28, 11],
    "solarfox": [10, 11]
}
game = "boulderdash"
for d in [10]:
    gen = ParamGenerator("./data/pca/{}/{}/".format(game,d), game, game_sizes[game][0], game_sizes[game][1])
    for i in range(1000):
        level = gen.generate([(d/10.0)], difficulty=True)
        #print(level)
print("Done")