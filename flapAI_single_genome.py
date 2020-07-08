import pygame
import random
import numpy as np
import neat     
pygame.init()

display_width = 330
display_height = 600
white = (255,255,255)
black = (0,0,0)
generation = 0

gameDisplay = pygame.display.set_mode((display_width,display_height))
clock = pygame.time.Clock()

def bird(x,y):
    pygame.draw.rect(gameDisplay,black,(x,y,30,30))
      
def poles(pos,num):
    pygame.draw.rect(gameDisplay, black,(pos,0,width_poleU,height_poleU+num))
    pygame.draw.rect(gameDisplay, black,(pos,height_poleU+100+num,width_poleU,display_height-height_poleU))

def game_over(num,pos,height_poleU):
    if (y<=height_poleU+num and x==pos-30) or (y>=height_poleU+90+num and x == pos-30):
        return True
        
    if y>display_height or y<0:
        return True
        
        
    return False
def distance(x,y,z,w):
    dist = (x-z)**2 + (y-w)**2
    return int(np.sqrt(dist))
    
def draw_distance(x,y,pos):
    arr = [] 
    arr.append(distance(x+15,y+15,pos,height_poleU+num))
    arr.append(distance(x+15,y+15,pos,height_poleU+num+100))        
    arr.append(distance(x+15,y+15,pos,height_poleU+50+num))
                
    pygame.draw.line(gameDisplay,black,(x+15,y+15),(pos,height_poleU+num))
    pygame.draw.line(gameDisplay, black,(x+15,y+15),(pos,height_poleU+100+num))
    pygame.draw.line(gameDisplay, black,(x+15,y+15),(pos,height_poleU+50+num))
    return(arr)
pos = 350
pos1 = 360
num = random.randint(100,200) 
height_poleU=150
width_poleU = 20
x = int(display_width*0.25)
y = display_height*0.4
 
dy = 0

def game(genomes,config):
    
    global x, y,pos,num,height_poleU,width_poleU,dy
    crashed = False

    nets = []
    birds = []
    x = int(display_width*0.25)
    y = display_height*0.4
    
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        birds.append(bird(x,y))

    
    global generation
    generation +=1

    while not crashed:
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
        y += dy
        pos -= 1
    
        output = net.activate(draw_distance(x,y,pos))
        i = output
        print(output)
        if int(i[0])==1:
            y -=1
        elif int(i[0]) == -1:
            y+=1
        for i in range(len(genomes)):
            if not game_over(num, pos, height_poleU):
                genomes[i][1].fitness += 0.02
            else:
                genomes[i][1].fitness -= 9
        
    
        gameDisplay.fill(white)
        bird(x,y)
        draw_distance(x,y,pos)
        
        poles(pos,num)
        
        if game_over(num,pos,height_poleU):
            crashed = True
            pos = 350
            break
        if pos == 0:
            pos = 390
            num = random.randint(100,200) 
            poles(pos,num)
    
        pygame.display.update()
    
        clock.tick(300)

    #pygame.quit()


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
    p.run(game, 40)