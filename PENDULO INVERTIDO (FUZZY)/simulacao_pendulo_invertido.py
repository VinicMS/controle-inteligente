#ESSE TA FUNCIONANDO MAIS OU MENOS (TREMENDO NO SETPOINT)

import pygame
import numpy as np
import pandas as pd
from scipy import linalg    

from skfuzzy import control as ctrl
import skfuzzy as fuzz
import matplotlib.pyplot as plt


class InvertedPendulum():
    # Initialize environment.
    def __init__(self, xRef = 0.0, randomParameters = False, randomSensor = False, randomActuator = False):
        # System parameters.
        self.tau = 0.01
        if not randomParameters:
            self.g = 9.8
            self.M = 1.0
            self.m = 0.1
            self.l = 0.5
        else:
            self.g = 9.8 + 0.098*np.random.randn()
            self.M = 1.0 + 0.1 *np.random.randn()
            self.m = 0.1 + 0.01*np.random.randn()
            self.l = 0.5 + 0.05*np.random.randn()
            
        self.xRef = xRef

        # Drawing parameters.
        self.cartWidth = 80
        self.cartHeight = 40
        self.pendulumLength = 200
        self.baseLine = 350
        self.screenWidth = 800
        self.screenHeight = 400
        
        # Variable to see if simulation ended.
        self.finish = False
        
        # Variable to see if there is randomness in the sensors and actuators.
        self.randomSensor   = randomSensor
        self.randomActuator = randomActuator
        
        # Create a random observation.
        self.reset()

        # Create screen.
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        pygame.display.set_caption('Inverted Pendulum')
        self.screen.fill('White')
        
        # Create a clock object.
        self.clock = pygame.time.Clock()
        pygame.display.update()

    # Close environment window.
    def close(self):
        pygame.quit()
        
    # Reset system with a new random initial position.
    def reset(self):
        self.observation = np.random.uniform(low = -0.05, high = 0.05, size = (4,))
        if self.randomSensor:
            return self.noise_sensors(self.observation.copy())
        else:
            return self.observation.copy()
    
    # Insert noise on the sensors.
    def noise_sensors(self, observation, noiseVar = 0.01):
        observation[0] = observation[0] + noiseVar*np.random.randn()
        observation[1] = observation[1] + noiseVar*np.random.randn()
        observation[2] = observation[2] + noiseVar*np.random.randn()
        observation[3] = observation[3] + noiseVar*np.random.randn()
        return observation
    
    # Insert noise on actuator.
    def noise_actuator(self, action, noiseVar = 0.01):
        action += noiseVar * np.random.randn()
        return action
    
    # Display object.
    def render(self):
        # Check for all possible types of player input.
        for event in pygame.event.get():
            # Command for closing the window.
            if (event.type == pygame.QUIT):
                pygame.quit()
                self.finish = True

                tabelaDf.to_csv('dados.csv', index=False, header=None)

                return None
            
            if (event.type == pygame.KEYDOWN):
                if (event.key == pygame.K_LEFT):
                    self.xRef -= 0.01
                    
                elif (event.key == pygame.K_RIGHT):
                    self.xRef += 0.01
                    
                elif (event.key == pygame.K_SPACE):
                    self.step(200*np.random.randn())
        
        # Apply surface over display.
        self.screen.fill('White')
        pygame.draw.line(self.screen, 'Black', (0, self.baseLine), (self.screenWidth, self.baseLine))
        
        # Get position for cart.
        xCenter = self.screenHeight + self.screenHeight * self.observation[0]
        xLeft   = xCenter - self.cartWidth//2
        # xRight  = xCenter + self.cartWidth//2
        
        # Get position for pendulum.
        pendX = xCenter +  self.pendulumLength * np.sin(self.observation[2])
        pendY = self.baseLine - self.pendulumLength * np.cos(self.observation[2])
        
        # Display objects.
        pygame.draw.line(self.screen,   'Green', (int(self.screenHeight + self.xRef * self.screenHeight), 0), (int(self.screenHeight + self.xRef * self.screenHeight), self.baseLine), width = 1)
        pygame.draw.rect(self.screen,   'Black', (xLeft, self.baseLine-self.cartHeight//2, self.cartWidth, self.cartHeight),  width = 0)
        pygame.draw.line(self.screen,   (100, 10, 10),   (xCenter, self.baseLine), (pendX, pendY), width = 6)
        pygame.draw.circle(self.screen, 'Blue',  (xCenter, self.baseLine), 10)
    
        # Draw all our elements and update everything.
        pygame.display.update()
        
        # Limit framerate.
        self.clock.tick(60)

    # Perform a step.
    def step(self, force):
        if self.randomActuator:
            force = self.noise_actuator(force)
        x1 = self.observation[0]
        x2 = self.observation[1]
        x3 = self.observation[2]
        x4 = self.observation[3]
        x4dot = (self.g * np.sin(x3) - np.cos(x3) * (force + self.m * self.l * x4**2 * np.sin(x3))/(self.M + self.m)) / (self.l * (4.0/3.0 - self.m * np.cos(x3)**2 / (self.M + self.m)))
        x2dot = (force + self.m * self.l * x4**2 * np.sin(x3))/(self.M + self.m) - self.m * self.l * x4dot * np.cos(x3) / (self.M + self.m)
        self.observation[0] = x1 + self.tau * x2
        self.observation[1] = x2 + self.tau * x2dot
        self.observation[2] = x3 + self.tau * x4
        self.observation[3] = x4 + self.tau * x4dot
        if self.randomSensor:
            return self.noise_sensors(self.observation.copy())
        else:
            return self.observation.copy()
        
# Parâmetros do sistema.
g = 9.8
M = 1.0
m = 0.1
l = 0.5
L = 2*l
I = m*L**2 / 12

# SENSORES.
# sensores[0]: posição.
# sensores[1]: velocidade.
# sensores[2]: ângulo.
# sensores[3]: velocidade angular.
# SETPOINT em env.xRef.

# Função de controle: Controle clássico.
def funcao_controle_1(sensores):
    
    #Polos com estabilidade fina para o disturbio com polos em p = (-80, -165, -1.5, -2) no Matlab
    K = np.array([-2761, -3273, -11460, -2352])
    
    #Polos com estabilidade fina para o disturbio com polos em P = [-4, -3.9719+j,-3.9719-j,-3.9719] no Matlab
    #Próximo do limiar de instabilidade (polos exatos com deslocamento no plano complexo)
    #K = np.array([-18.5846, -18.1254, -88.7634, -22.9593])
                                                                                 
    #ganhos obtidos com os polos estipulados por análise da reação ao degrau
    K = K*(-1)#operação da fórmula
    
    acao = np.array(((sensores[0]-env.xRef)*K[0]+sensores[1]*K[1]+sensores[2]*K[2]+sensores[3]*K[3]))
    
    print("X: %.2f, X_P: %.2f, T: %.2f, T_P: %.2f C: %.2f" % 
        (env.xRef-sensores[0], sensores[1], sensores[2], sensores[3], acao))

    return acao

#Definição dos parâmetros para construir o controlador Fuzzy

x = np.linspace(-1, 1, 1000)
x_p = np.linspace(-75, 75, 2000)
theta = np.linspace(-1, 1, 1000)
theta_p = np.linspace(-20, 20, 1000)

forca = np.linspace(-1000, 1000, 1000)

X = ctrl.Antecedent(x, 'Posição')
X_P = ctrl.Antecedent(x_p, 'Velocidade')
THETA = ctrl.Antecedent(theta, 'Angulo')
THETA_P = ctrl.Antecedent(theta_p, 'Velocidade Angular')

FORCA = ctrl.Consequent(forca, 'Força', defuzzify_method='centroid')
#defuzzify_method = 'centroid'
#defuzzify_method = 'bisector'
#defuzzify_method = 'mom'
#defuzzify_method = 'som'
#defuzzify_method = 'lom'

K_posicao = 70
K_velocidade = 0.9
K_angulo = 0.8
K_velocidade_angular = 0.4

K_f = 1.5

X['MuitoNegativo'] = fuzz.trapmf(x, [-1, -1, -0.75,  -0.5])
X['PoucoNegativo'] = fuzz.trimf( x, [-0.75, -0.325, -0])
X['Zero']          = fuzz.trimf( x, [-0.4, 0.0, 0.4])
X['PoucoPositivo'] = fuzz.trimf( x, [0, 0.325, 0.75])
X['MuitoPositivo'] = fuzz.trapmf(x, [0.5, 0.75, 1, 1])

X_P['Negativo'] = fuzz.trimf(x_p, [-50, -47, 0])
X_P['Zero']     = fuzz.trimf(x_p, [-0.1, 0, 0.1])
X_P['Positivo'] = fuzz.trimf(x_p, [0, 47, 50])

THETA['MuitoNegativo'] = fuzz.trapmf(theta, [-1, -1, -0.75, -0.5])
THETA['MedioNegativo'] = fuzz.trimf( theta, [-0.75, -0.5, -0.25])
THETA['PoucoNegativo'] = fuzz.trimf( theta, [-0.50, -0.25, 0])
THETA['Zero']          = fuzz.trapmf(theta, [-0.2, -0.1, 0.1, 0.2])
THETA['PoucoPositivo'] = fuzz.trimf( theta, [0 , 0.25, 0.5])
THETA['MedioPositivo'] = fuzz.trimf( theta, [0.25, 0.5, 0.75])
THETA['MuitoPositivo'] = fuzz.trapmf(theta, [0.5, 0.75, 1, 1])

THETA_P['MuitoNegativo'] = fuzz.trapmf(theta_p, [-20,-20,-12,-7])
THETA_P['PoucoNegativo'] = fuzz.trimf( theta_p, [-8,-5,-2])
THETA_P['Zero']          = fuzz.trimf( theta_p, [-3, 0, 3])
THETA_P['PoucoPositivo'] = fuzz.trimf( theta_p, [2, 5, 8])
THETA_P['MuitoPositivo'] = fuzz.trapmf(theta_p, [7,12,20,20])

FORCA['ExtremoNegativo'] = fuzz.trapmf(forca, [-1000, -1000, -900, -600])
FORCA['MuitoNegativo']   = fuzz.trimf( forca, [-800, -600, -400])
FORCA['MedioNegativo']   = fuzz.trimf( forca, [-600, -400, -200])
FORCA['PoucoNegativo']   = fuzz.trimf( forca, [-400, -200, -200])
FORCA['Zero']            = fuzz.trimf( forca, [-100, 0, 100])
FORCA['PoucoPositivo']   = fuzz.trimf( forca, [200, 200, 400])
FORCA['MedioPositivo']   = fuzz.trimf( forca, [200, 400, 600])
FORCA['MuitoPositivo']   = fuzz.trimf( forca, [400, 600, 800])
FORCA['ExtremoPositivo'] = fuzz.trapmf(forca, [600, 900, 1000, 1000])

#FORCA.view()

#REGRAS X/X_P
#Para as interpretações de posição a ação de controle é inversa à variação (erro)

RegraX_1 = ctrl.Rule( X['MuitoNegativo']     & X_P['Negativo'], FORCA['ExtremoPositivo'])
RegraX_2 = ctrl.Rule( X['PoucoNegativo']     & X_P['Negativo'], FORCA['PoucoPositivo'])
RegraX_3 = ctrl.Rule( X['Zero']              & X_P['Negativo'], FORCA['PoucoPositivo'])
RegraX_4 = ctrl.Rule( X['Zero']              & X_P['Zero']    , FORCA['Zero'])
RegraX_5 = ctrl.Rule( X['Zero']              & X_P['Positivo'], FORCA['PoucoNegativo'])
RegraX_6 = ctrl.Rule( X['PoucoPositivo']     & X_P['Positivo'], FORCA['PoucoNegativo'])
RegraX_7 = ctrl.Rule( X['MuitoPositivo']     & X_P['Positivo'], FORCA['ExtremoNegativo'])

RegraX_8 = ctrl.Rule( X['PoucoNegativo']     & X_P['Positivo'], FORCA['PoucoPositivo'])
RegraX_9 = ctrl.Rule( X['PoucoPositivo']     & X_P['Negativo'], FORCA['PoucoNegativo'])
RegraX_10 = ctrl.Rule( X['PoucoNegativo']    & X_P['Zero'], FORCA['Zero'])
RegraX_11 = ctrl.Rule( X['PoucoPositivo']    & X_P['Zero'], FORCA['Zero'])

RegraF_1 = ctrl.Rule( X['Zero']                            & THETA['Zero'], FORCA['Zero'])
RegraF_2 = ctrl.Rule( X['Zero']          & X_P['Zero']     & THETA['Zero'], FORCA['Zero'])
RegraF_3 = ctrl.Rule( X['PoucoPositivo'] & X_P['Positivo'] & THETA['Zero'], FORCA['Zero'])
RegraF_4 = ctrl.Rule( X['PoucoNegativo'] & X_P['Negativo'] & THETA['Zero'], FORCA['Zero'])
RegraF_5 = ctrl.Rule( X['PoucoPositivo'] & X_P['Negativo'] & THETA['Zero'], FORCA['Zero'])
RegraF_6 = ctrl.Rule( X['PoucoNegativo'] & X_P['Positivo'] & THETA['Zero'], FORCA['Zero'])

#REGRAS THETA/THETA_P
RegraT_1 = ctrl.Rule( THETA['MuitoNegativo'] & THETA_P['MuitoNegativo'], FORCA['ExtremoNegativo'])
RegraT_2 = ctrl.Rule( THETA['MuitoNegativo'] & THETA_P['PoucoNegativo'], FORCA['ExtremoNegativo'])
RegraT_3 = ctrl.Rule( THETA['MuitoNegativo'] & THETA_P['Zero']         , FORCA['MuitoNegativo'])
RegraT_4 = ctrl.Rule( THETA['MuitoNegativo'] & THETA_P['PoucoPositivo'], FORCA['MedioNegativo'])
RegraT_5 = ctrl.Rule( THETA['MuitoNegativo'] & THETA_P['MuitoPositivo'], FORCA['PoucoNegativo'])

RegraT_6 = ctrl.Rule( THETA['MedioNegativo'] & THETA_P['MuitoNegativo'], FORCA['ExtremoNegativo'])
RegraT_7 = ctrl.Rule( THETA['MedioNegativo'] & THETA_P['PoucoNegativo'], FORCA['MuitoNegativo'])
RegraT_8 = ctrl.Rule( THETA['MedioNegativo'] & THETA_P['Zero']         , FORCA['MedioNegativo'])
RegraT_9 = ctrl.Rule( THETA['MedioNegativo'] & THETA_P['PoucoPositivo'], FORCA['PoucoNegativo'])
RegraT_10 = ctrl.Rule(THETA['MedioNegativo'] & THETA_P['MuitoPositivo'], FORCA['Zero'])

RegraT_11 = ctrl.Rule(THETA['PoucoNegativo'] & THETA_P['MuitoNegativo'], FORCA['MuitoNegativo'])
RegraT_12 = ctrl.Rule(THETA['PoucoNegativo'] & THETA_P['PoucoNegativo'], FORCA['MedioNegativo'])
RegraT_13 = ctrl.Rule(THETA['PoucoNegativo'] & THETA_P['Zero']         , FORCA['PoucoNegativo'])
RegraT_14 = ctrl.Rule(THETA['PoucoNegativo'] & THETA_P['PoucoPositivo'], FORCA['Zero'])
RegraT_15 = ctrl.Rule(THETA['PoucoNegativo'] & THETA_P['MuitoPositivo'], FORCA['PoucoPositivo'])

RegraT_16 = ctrl.Rule(THETA['Zero']          & THETA_P['MuitoNegativo'], FORCA['MedioNegativo'])
RegraT_17 = ctrl.Rule(THETA['Zero']          & THETA_P['PoucoNegativo'], FORCA['PoucoNegativo'])
RegraT_18 = ctrl.Rule(THETA['Zero']          & THETA_P['Zero']         , FORCA['Zero'])
RegraT_19 = ctrl.Rule(THETA['Zero']          & THETA_P['PoucoPositivo'], FORCA['PoucoPositivo'])
RegraT_20 = ctrl.Rule(THETA['Zero']          & THETA_P['MuitoPositivo'], FORCA['MedioPositivo'])

RegraT_21 = ctrl.Rule(THETA['PoucoPositivo'] & THETA_P['MuitoNegativo'], FORCA['PoucoNegativo'])
RegraT_22 = ctrl.Rule(THETA['PoucoPositivo'] & THETA_P['PoucoNegativo'], FORCA['Zero'])
RegraT_23 = ctrl.Rule(THETA['PoucoPositivo'] & THETA_P['Zero']         , FORCA['PoucoPositivo'])
RegraT_24 = ctrl.Rule(THETA['PoucoPositivo'] & THETA_P['PoucoPositivo'], FORCA['MedioPositivo'])
RegraT_25 = ctrl.Rule(THETA['PoucoPositivo'] & THETA_P['MuitoPositivo'], FORCA['MuitoPositivo'])

RegraT_26 = ctrl.Rule(THETA['MedioPositivo'] & THETA_P['MuitoNegativo'], FORCA['Zero'])
RegraT_27 = ctrl.Rule(THETA['MedioPositivo'] & THETA_P['PoucoNegativo'], FORCA['PoucoPositivo'])
RegraT_28 = ctrl.Rule(THETA['MedioPositivo'] & THETA_P['Zero']         , FORCA['MedioPositivo'])
RegraT_29 = ctrl.Rule(THETA['MedioPositivo'] & THETA_P['PoucoPositivo'], FORCA['MuitoPositivo'])
RegraT_30 = ctrl.Rule(THETA['MedioPositivo'] & THETA_P['MuitoPositivo'], FORCA['ExtremoPositivo'])

RegraT_31 = ctrl.Rule(THETA['MuitoPositivo'] & THETA_P['MuitoNegativo'], FORCA['PoucoPositivo'])
RegraT_32 = ctrl.Rule(THETA['MuitoPositivo'] & THETA_P['PoucoNegativo'], FORCA['MedioPositivo'])
RegraT_33 = ctrl.Rule(THETA['MuitoPositivo'] & THETA_P['Zero']         , FORCA['MuitoPositivo'])
RegraT_34 = ctrl.Rule(THETA['MuitoPositivo'] & THETA_P['PoucoPositivo'], FORCA['ExtremoPositivo'])
RegraT_35 = ctrl.Rule(THETA['MuitoPositivo'] & THETA_P['MuitoPositivo'], FORCA['ExtremoPositivo'])

compilado_regras = ctrl.ControlSystem([
    RegraX_1, RegraT_1,
    RegraX_2, RegraT_2,
    RegraX_3, RegraT_3,
    RegraX_4, RegraT_4,
    RegraX_5, RegraT_5,
    RegraX_6, RegraT_6,
    RegraX_7, RegraT_7,
    RegraX_8, RegraT_8,
    RegraX_9, RegraT_9,
    RegraF_1, RegraT_10,
    RegraF_2, RegraT_11,
    RegraF_3, RegraT_12,
    RegraF_4, RegraT_13,
    RegraF_5, RegraT_14,
    RegraF_6, RegraT_15,
    RegraX_10,RegraT_16,
    RegraX_11,RegraT_17,
              RegraT_18,
              RegraT_19,
              RegraT_20,
              RegraT_21,
              RegraT_22,
              RegraT_23,
              RegraT_24,
              RegraT_25,
              RegraT_26,
              RegraT_27,
              RegraT_28,
              RegraT_29,
              RegraT_30,
              RegraT_31,
              RegraT_32,
              RegraT_33,
              RegraT_34,
              RegraT_35])

Controle = ctrl.ControlSystemSimulation(compilado_regras)

# Função de controle Fuzzy.

def funcao_controle_2(sensores):
    posicao = env.xRef - sensores[0]

    Controle.input['Posição'] = posicao*K_posicao
    ind_posicao = posicao
    Controle.input['Velocidade'] = sensores[1]*K_velocidade
    ind_velocidade = sensores[1]
    Controle.input['Angulo'] = sensores[2]*K_angulo
    ind_angulo = sensores[2]
    Controle.input['Velocidade Angular'] = sensores[3]*K_velocidade_angular
    ind_velocidade_angular = sensores[3]

    Controle.compute() #defuzzify_method = 'centroid'
    
    # if abs(posicao)<0.02 and abs(ind_velocidade)<0.01 and abs(ind_angulo)<0.001:
    #     acao = 0
    # else:
    #     acao = Controle.output['Força']*K_f

    acao = Controle.output['Força']*K_f
    
    ind_acao = acao
    
    print("X: %.2f, X_P: %.2f, T: %.2f, T_P: %.2f C: %.2f" % 
        (ind_posicao, ind_velocidade, ind_angulo, ind_velocidade_angular, ind_acao))

    return acao
                
# Função de controle.
def funcao_controle_3(sensores):
    # Controle intuitivo.
    # Obtém valor do ângulo.
    angulo = sensores[2]
    if (angulo > 0):
        acao = +1.0
    # Se o pêndulo está caindo para a esquerda, movemos o carro para a esquerda.
    else:
        acao = -1.0

    print("X: %.2f, X_P: %.2f, T: %.2f, T_P: %.2f C: %.2f" % 
        (env.xRef-sensores[0], sensores[1], sensores[2], sensores[3], acao))

    return acao

# Cria o ambiente de simulação.
env = InvertedPendulum(0.50)

grafico_posicao = []
grafico_acao = []

# Reseta o ambiente de simulação.
sensores = env.reset()

tabela = [(0, 0, 0, 0, 0)]
tabelaDf = pd.DataFrame(tabela)
k = 0

while True:
    # Renderiza o simulador.
    env.render()
    if env.finish:
        break
    
    # Calcula a ação de controle.
    acao = funcao_controle_2(sensores)  # É ESSA A FUNÇÃO QUE VOCÊS DEVEM PROJETAR.
    
    grafico_posicao.append(env.xRef - sensores[0])
    grafico_acao.append(acao)
    
    # Aplica a ação de controle.
    sensores = env.step(acao)

    tabelaDf.loc[k] = [env.xRef-sensores[0], sensores[1], sensores[2], sensores[3], acao]
    k = k + 1
    
env.close()

plt.plot(grafico_posicao)
#plt.plot(grafico_acao)