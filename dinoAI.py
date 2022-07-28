from math import exp
import pygame
import os
import random
import time
from sys import exit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

weightQtd = 35

class KeyRicClassifier(KeyClassifier):
    def __init__(self, weight):
        self.weight = weight

    def keySelector(self, obDistance, obHeight, scSpeed, obWidth, diHeight, obDistance2, obHeight2, obWidth2):
        
        op = self.neuronsOp([obDistance, obHeight, obWidth, scSpeed, diHeight], [5, 5, 2], [sigmoid, degrau]) # Total = 35
        
        if op[0] == 1:
            return "K_UP"
        elif op[1] == 1:
            return "K_DOWN"
        return "K_NO"

    def neuronsOp(self, value, neurons, func_list):
        newNeurons = value.copy()
        prevNeurons = []
        position = 0
        for n in range(len(neurons)):
            if n == 0:
                continue
            func = func_list[n-1]
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
        obDistance2 = 0
        obHeight2 = 0
        obWidth2 = 0
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            obDistance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]
            obWidth = obstacles[0].rect.width
        if len(obstacles) > 1:
            xy2 = obstacles[1].getXY()
            obDistance2 = xy2[0]
            obHeight2 = obstacles[1].getHeight()
            obWidth2 = obstacles[1].rect.width

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            userInput = aiPlayer.keySelector(obDistance, obHeight, game_speed, obWidth, player.getXY()[1], obDistance2, obHeight2, obWidth2)

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

def generate_childs(weight_list, crossoverQtd, bestValue, mutationRate):
    auxCrossover = []
    listSize = len(weight_list)
    for it in range(listSize):
        weight1 = weight_list[it][1]
        auxCrossover.append(weight1)
        for it2 in range(listSize):
            if it == it2:
                continue
            weight2 = weight_list[it2][1]
            newMutationRate = 0.2*(1 - (weight_list[it][0] + weight_list[it2][0])/(2*bestValue)) + 0.01 * mutationRate
            auxCrossover += crossover(weight1, weight2, crossoverQtd, newMutationRate)
    return auxCrossover
    

# Mutation

def mutation(state, mutationRate):
    aux = state.copy()
    state_size = len(state)
    for it in range(state_size):
        if random.random() < mutationRate:
            aux[it] +=  random.randint(-50,50)
            if aux[it] < -1000:
                aux[it] = -1000
            if aux[it] > 1000:
                aux[it] = 1000
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

    #### Inicializar tabela de valores
    f = open("table.csv", "w")
    f.write("Gen;Time;Best Score;First Place Generation;Second Place Generation;Third Place Generation\n")
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
            values_list .append([random.randint(-1000,1000) for col in range(weightQtd)])
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
    saveCSV(weights, generation, time.time() - start, best_value)
    ####

    #### Aplicar metaheuristica em busca de encontrar melhores pesos 
    generation+=1
    mutationRate = 0
    while end - start <= max_time:
        count = 0
        
        newWeights = generate_childs(weights[0 : 5], 3, best_value, mutationRate)
        weights.clear()

        for s in newWeights:
            count += 1
            aiPlayer = KeyRicClassifier(s)
            _, value = manyPlaysResults(plays)
            if value > best_value:
                best_value = value
                best_state = s
            weights.append([value, s])
        end = time.time()
        weights.sort(reverse=True)

        if weights[0][0] >= best_value:
            mutationRate = 0
        else:
            mutationRate += 1
        saveWeights(weights, generation, time.time() - start)
        saveCSV(weights, generation, time.time() - start, best_value)
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

def saveCSV(weights, gen, time, best_score):
    f = open("table.csv", "a")
    f.write(str(gen) + ";" + str(time) + ';' + str(best_score) + ';' + str(weights[0][0]) + ';' + str(weights[1][0]) + ';' + str(weights[2][0]) + "\n")
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

def testValue():
    global aiPlayer
    test_state = [986, 843, -229, -615, -581, 393, -511, 590, -1000, -343, -741, -251, -430, 303, 578, -22, 908, -108, -380, -149, -693, 556, 516, 313, -770, 11, -475, -939, -868, -200, -680, -701, 388, 438, 669]
    aiPlayer = KeyRicClassifier(test_state)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)

def printBoxplot():
    neuralResult = [2116.75, 2147.0, 2053.25, 1863.5, 2105.75, 2016.75, 2007.5, 2120.75, 1905.75, 1808.25, 2118.0, 2091.0, 2004.75, 2136.0, 2012.75, 1862.25, 1997.75, 1843.5, 2105.0, 1801.0, 2006.25, 1975.0, 1901.25, 1906.75, 1906.5, 1907.0, 2190.0, 2013.5, 1906.5, 2105.0]
    simpleResult = [1214.0, 759.5, 1164.25, 977.25, 1201.0, 930.0, 1427.75, 799.5, 1006.25, 783.5, 728.5, 419.25, 1389.5, 730.0, 1306.25, 675.5, 1359.5, 1000.25, 1284.5, 1350.0, 751.0, 1418.75, 1276.5, 1645.75, 860.0, 745.5, 1426.25, 783.5, 1149.75, 1482.25]
    dataFrame = np.array([neuralResult, simpleResult])
    df = pd.DataFrame(data=dataFrame.transpose(), columns=["Rede Neural", "KeySimplestClassifier"])

    sns.boxplot(data=df)
    plt.show()

main()
#testValue()
#printBoxplot()

# [986, 843, -229, -615, -581, 393, -511, 590, -1000, -343, -741, -251, -430, 303, 578, -22, 908, -108, -380, -149, -693, 556, 516, 313, -770, 11, -475, -939, -868, -200, -680, -701, 388, 438, 669]

# [2116.75, 2147.0, 2053.25, 1863.5, 2105.75, 2016.75, 2007.5, 2120.75, 1905.75, 1808.25, 2118.0, 2091.0, 2004.75, 2136.0, 2012.75, 1862.25, 1997.75, 1843.5, 2105.0, 1801.0, 2006.25, 1975.0, 1901.25, 1906.75, 1906.5, 1907.0, 2190.0, 2013.5, 1906.5, 2105.0] 1997.8333333333333 108.91197388513145 1888.9213594482019