import pygame
import random
import numpy as np
import neat     
pygame.init()
clock = pygame.time.Clock()

display_width = 330
display_height = 600
gameDisplay = pygame.display.set_mode((display_width,display_height))

white = (255,255,255)
black = (0,0,0)
generation = 0
g_size = 50

def bird(x,y):
    pygame.draw.rect(gameDisplay,black,(x,y,30,30))
#draws the obstacles     
def poles(pos,num):
    pygame.draw.rect(gameDisplay, black,(pos,0,width_poleU,height_poleU+num))
    pygame.draw.rect(gameDisplay, black,(pos,height_poleU+g_size+num,width_poleU,display_height-height_poleU))

def game_over(num,pos,height_poleU,y):
    if (y<=height_poleU+num and x==pos-30) or (y>=height_poleU+g_size+num and x == pos-30):
        return True
        
    if y>display_height or y<0:
        return True 
    return False

def distance(x,y,z,w):
    dist = (x-z)**2 + (y-w)**2
    return int(np.sqrt(dist))
    
def get_distance(x,y,pos):
    arr = [] 
    arr.append(distance(x+15,y+15,pos,height_poleU+num))
    arr.append(distance(x+15,y+15,pos,height_poleU+num+g_size))        
    arr.append(distance(x+15,y+15,pos,height_poleU+g_size/2+num))         
    return(arr)

def draw_distance(x,y,pos):
    pygame.draw.line(gameDisplay,black,(x+15,y+15),(pos,height_poleU+num))
    pygame.draw.line(gameDisplay, black,(x+15,y+15),(pos,height_poleU+g_size+num))
    pygame.draw.line(gameDisplay, black,(x+15,y+15),(pos,height_poleU+g_size/2+num))


pos = 350
num = random.randint(100,500) 
height_poleU=10
width_poleU = 20
x = int(display_width*0.25)
y = display_height*0.4 
dy = 0

def game(genomes,config):
    
    global x, y,pos,num,height_poleU,width_poleU,dy

    nets = []
    birds = []
    number = random.randint(200, 500)
    y_arr = [number]*len(genomes)

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
    for i in range(len(y_arr)):
        birds.append(bird(x,y_arr[i]))

    global generation
    generation +=1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed =True
                sys.exit(0)
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    dy = -1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    dy = +1
        #y += dy
        pos -= 1
        gameDisplay.fill(white)

        for index,birdd in enumerate(birds):

            output = nets[index].activate(get_distance(x,y_arr[index],pos))
            if int(output[0])==1:
                y_arr[index] =y_arr[index]-3
            elif int(output[0])== -1:
                y_arr[index] =y_arr[index]+3
        
        bird_remain = 0

        for i in range(len(genomes)):
            if not game_over(num, pos, height_poleU,y_arr[i]):
                genomes[i][1].fitness += 0.02
                draw_distance(x,y_arr[i],pos)
                bird_remain +=1
            else:
                genomes[i][1].fitness -= 3
        
        if bird_remain == 0:
            break
           
        for i in range(len(y_arr)):
            bird(x,y_arr[i])

        for i in range(len(y_arr)):
            if game_over(num,pos,height_poleU,y_arr[i]):
                break
                pos = 350
                num = random.randint(100,500) 


        if pos == 0:
            pos = 390
            num = random.randint(100,500) 
        poles(pos,num)
        
        pygame.display.update()
        clock.tick(300)

    

#default setup for NEAT
if __name__ == "__main__":
    # Set configuration file
    config_path = "config.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    p.run(game, 100)