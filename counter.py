import pygame, os


def count_files():

    pygame.init()

    font = pygame.font.SysFont('comicsans', 60, True)
    win = pygame.display.set_mode((450, 400))
    pygame.display.set_caption("Directory Size Counter by Sanay N.")

    run = True
    while run:

        win.fill((18, 52, 86))
        length = len(os.listdir("data_set/sanay2"))
        text = font.render(str(length), True, (255, 255, 255))

        win.blit(text, (195, 200))
        pygame.display.update()
        
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False
    
    pygame.quit()

count_files()