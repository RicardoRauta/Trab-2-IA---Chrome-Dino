from math import exp, tanh
from unittest import result
import pygame
import os
import random
import time
from sys import exit
import matplotlib.pyplot as plt

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
GRAPH = False

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
if GRAPH:
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass

def sigmoid(x):
    if x >= 0:
        z = exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = exp(x)
        sig = z / (1 + z)
        return sig
def degrau(x):
    if x > 0:
        return 1
    return 0

weightQtd = 85

class KeyRicClassifier(KeyClassifier):
    def __init__(self, weight):
        self.weight = weight

    def keySelector(self, obDistance, obHeight, scSpeed, obWidth, diHeight, obType):
        isFlying = 0
        if obHeight > 50:
            isFlying = 1
        op = self.neuronsOp([obDistance, obHeight, obWidth, scSpeed, diHeight, isFlying], [6, 5, 5, 5, 1], degrau) # Total = 147
        
        if op[0] > 0:
            return "K_UP"
        else:
            return "K_DOWN"
        return "K_NO"

    def neuronsOp(self, value, neurons, func):
        newNeurons = value.copy()
        prevNeurons = []
        position = 0
        for n in range(len(neurons)):
            if n == 0:
                continue
            prevNeurons = newNeurons.copy()
            newNeurons.clear()
            for it in range(neurons[n]):
                sum = 0
                for it2 in range(neurons[n-1]):
                    sum += prevNeurons[it2] * self.weight[position]
                    position += 1
                newNeurons.append(func(sum))
        return newNeurons

    def updateWeight(self, weight):
        self.weight = weight


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if GRAPH:
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if GRAPH:
            SCREEN.fill((255, 255, 255))

        obDistance = 1500
        obHeight = 0
        obType = 2
        obWidth = 0
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            obDistance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]
            obWidth = obstacles[0].rect.width

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            userInput = aiPlayer.keySelector(obDistance, obHeight, game_speed, obWidth, player.getXY()[1], obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            onlyBird = False
            if onlyBird:
                obstacles.append(Bird(BIRD))
            elif random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        if GRAPH:
            player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            if GRAPH:
                obstacle.draw(SCREEN)

        if GRAPH:
            background()

            cloud.draw(SCREEN)
        cloud.update()

        score()

        if GRAPH:
            clock.tick(60)#60
        if GRAPH:
            pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                if GRAPH:
                    pygame.time.delay(2000)#2000
                death_count += 1
                return points

# Genetic

def generate_childs(weight_list, crossoverQtd, bestValue):
    auxCrossover = []
    listSize = len(weight_list)
    for it in range(listSize):
        weight1 = weight_list[it][1]
        auxCrossover.append(weight1)
        for it2 in range(listSize):
            if it == it2:
                continue
            weight2 = weight_list[it2][1]
            mutationRate = 0.33*(1 - (weight_list[it][0] + weight_list[it2][0])/(2*bestValue))
            auxCrossover += crossover(weight1, weight2, crossoverQtd, mutationRate)
    return auxCrossover
    

# Mutation

def mutation(state, mutatationRate):
    aux = state.copy()
    if mutatationRate > 0.5:
        mutatationRate += 0.5
    if mutatationRate < 0.1:
        mutatationRate += 0.05
    state_size = len(state)
    for it in range(state_size):
        rand = random.randint(0, 100)
        if rand < mutatationRate*100:
            aux[it] +=  random.randint(-50, 50)/100
            if aux[it] < -1:
                aux[it] += 1
            if aux[it] > 1:
                aux[it] -= 1
    return aux
        
# Crossover

def crossover(state1, state2, childrensQtd, mutationRate):
    childrens = []
    for it in range(childrensQtd):
        randPos = random.randint(0, len(state1))
        newState = state1[:randPos] + state2[randPos:]
        childrens.append(mutation(newState, mutationRate))
    return childrens

def run(max_time, initial_value):
    global aiPlayer

    #### Inicializar arquivo de log vazio
    f = open("log.txt", "w")
    f.write("")
    f.close()
    ####

    plays = 3
    start = time.time()
    weights = []
    end = 0
    generation = 1
    best_value = 0
    
    #### Testar 30 listas de pesos aleatórias
    initial_size = 30
    values_list = []
    count = 0
    for it in range(initial_size):
            values_list .append([random.randint(-100, 100)/100 for col in range(weightQtd)])
    for newWeights in values_list:
        count += 1
        aiPlayer = KeyRicClassifier(newWeights)
        _, value = manyPlaysResults(plays)
        if value > best_value:
            best_value = value
            best_state = newWeights
        #print(generation, count, value)
        weights.append([value, newWeights])
    weights.sort(reverse=True)
    saveWeights(weights, generation, time.time() - start)
    ####

    #### Aplicar metaheuristica em busca de encontrar melhores pesos 
    generation+=1
    while end - start <= max_time:
        count = 0
        
        #print("Time: ", time.time() - start)
        newWeights = generate_childs(weights[0 : 5], 3, best_value)
        weights.clear()

        for s in newWeights:
            count += 1
            aiPlayer = KeyRicClassifier(s)
            _, value = manyPlaysResults(plays)
            if value > best_value:
                best_value = value
                best_state = s
            #print(generation, count, value)
            weights.append([value, s])
        end = time.time()
        weights.sort(reverse=True)
        saveWeights(weights, generation, time.time() - start)
        generation+=1
    ####
    
    return best_state, best_value

def saveWeights(weights, gen, time):
    f = open("log.txt", "a")
    f.write("Generation: " + str(gen) + "\n")
    f.write("Time: " + str(time) + "\n\n")
    it = 0
    for weight in weights:
        it+=1
        f.write(str(weight) + "\n")
        if it == 10:
            break
    f.write("\n\n\n")
    f.close()


from scipy import stats
import numpy as np


def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


def main():
    global aiPlayer
    initial_state = []
    best_state, best_value = run(24*60*60, initial_state) # rodar por 24 horas
    aiPlayer = KeyRicClassifier(best_state)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)
    f = open("log.txt", "a")
    f.write("Result: \n" + str(res) + "\nMean: " + str(npRes.mean()) + "\nStd: " + str(npRes.std()) + "\nValue: " + str(value))


main()