import pygame as pg
from random import random
from time import time
import tensorflow as tf

pg.init()

display = pg.display.set_mode((600, 800))

G = 25
JUMP_FORCE = 400

deltaT = 0


def gen_mutant(parent_model, mutation_rate):
    new_weights = parent_model.get_weights()
    for weight_array in new_weights:
        num_weights = weight_array.size
        num_weights_modified = np.random.binomial(num_weights, mutation_rate)
        for i in range(num_weights_modified):
            modify_weights(weight_array)
    mutant = tf.keras.models.clone_model(parent_model)
    mutant.set_weights(new_weights)
    return mutant

def modify_weights(weight_val):
    if np.isscalar(weight_val):
        np.random.normal(weight_val, abs(weight_val / 2))
    else:
        array_n = random.randint(0, len(weight_val)-1)
        modify_weights(weight_val[array_n])


class Blob:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.motion = pg.math.Vector2(0, 0)

    def move(self):
        self.motion.y += G*deltaT
        self.y += self.motion.y*deltaT
        self.x += self.motion.x*deltaT

    def think(self):
        if self.y >= 600:
            print("JUMP!")
            self.motion.y -= JUMP_FORCE

    def draw(self, surface):
        pg.draw.rect(surface, (255, 255, 255), (self.x, self.y, 10, 10))


class World:
    def __init__(self):
        self.blobs = []
    
    def add_blob(self, blob):
        self.blobs.append(blob)

    def add_blobs(self, blobs):
        self.blobs.extend(blobs)

    def draw(self, surface):
        for i in self.blobs:
            i.draw(surface)

    def move(self):
        for i in self.blobs:
            i.move()

    def think(self):
        for i in self.blobs:
            i.think()


world = World()

world.add_blob(Blob(50, 200))

while True:
    frameStartTime = time()
    display.fill((0, 0, 0))
    
    world.draw(display)
    world.move()
    world.think()

    pg.display.update()
    deltaT = time() - frameStartTime
