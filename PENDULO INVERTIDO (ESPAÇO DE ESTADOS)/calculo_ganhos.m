% Matrizes para o modelo no espaço de estados
A=[0 1 0 0;0 0 -0.71707317 0; 0 0 0 1;0 0 15.77560976 0];
B=[0;0.975609755;0;-1.463414634];
C=[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
D=[0; 0; 0; 0];

sysSS_ma = ss(A, B, C, D); %objeto no espaço de estados

%IMPLEMENTAÇÕES DE MÉTODOS DE DEFINIÇÃO DA MATRIZ DE GANHO K

co = ctrb(sysSS_ma); %matriz de controlabilidade
controlabilidade_reduzida = rank(co); %verifica se a dimensão de A é igual 
                                      %à pose da matriz de controlabilidade
                                      %como posto = 4 e A(4x4), é
                                      %controlavel.

%LQR

K1=lqr(A,B,[100 0 0 0;0 1 0 0; 0 0 1 0; 0 0 0 1],100);
A_lqr1=A-B*K1; %matriz A para malha fechada

K2=lqr(A,B,[0.1 0 0 0;0 1 0 0; 0 0 1 0; 0 0 0 1],0.1); 
A_lqr2=A-B*K2; %matriz A para malha fechada

% Seguindo indicacoes do site com a modelagem e desenvolvimento:
%https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace

K3=lqr(A,B,[5000 0 0 0;0 0 0 0; 0 0 100 0; 0 0 0 0],1);
A_lqr3=A-B*K3; %matriz A para malha fechada

%Place (polos)
p=[-80,-165,-1.5,-2]; %fino pra caramba

%p=[-80,-125,-2.0-0.25j,-2.0+0.25j];%Polos obtidos empiricamente para atingir um controle 
                     %estável a distúrbios com conjugados

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AQUI FOI A QUE DEU CERTO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K4=place(A,B,p);
A_pl=A-B*K4; %matriz A para malha fechada

[Num,Den] = ss2tf(A, B, C, D); %conversão do espaço de estados para o
                               %dominio da frequencia com a funcao de
                               %transferencia

Num1 = [0         0    0.9756    0.0000  -14.3415];
Num2 = [0    0.9756         0  -14.3415         0];
Num3 = [0         0   -1.4634         0         0];
Num4 = [0   -1.4634         0    0.0000         0];
                               
%sys = tf(Num, Den) % deu errado pois Num é vetor (matriz) e Den é linha
sys1 = tf(Num1,Den)
sys2 = tf(Num2,Den)
sys3 = tf(Num3,Den)
sys4 = tf(Num4,Den)

sysT_multiplicacao = sys1*sys2*sys3*sys4
sysT_soma = sys1 + sys2 + sys3 + sys4

%rlocus(sysT); %análise gráfica para o lugar das raízes

%roots(Den); %análise dos polos pela função de transferência

E = eig(A); %autovalor, visualiza os pontos problemáticos para a estabilidade
            %do sitema

P = [-80, -3.9719+j,-3.9719-j,-80.9719]; %valor desejado para os autovalores em malha fechada
                            %para atender a condição de estabilidade do sistema
                              
K = place(A, B, P); %obter uma matriz de ganho K que modificará os autovalores
                    %originais (eig(A)) para atingir os desejados (P)
          
A_pl_calculado = A - B*K; %Matriz A em Malha Fechada
Ecl = eig(A_pl_calculado); %verificacao do autovelor da nova matriz (tem que ser 
                           %estável
                
sysSS_mf = ss(A_pl_calculado, B, C, D); %Criando o sistema em malha fechada  
                                      %no espaço de estados (com A revisado)

sysSS_mf_empirico = ss(A_pl, B, C, D);
    
%step(sysSS_ma);
step(sysSS_mf);
step(sysSS_mf_empirico);
%plot(sysSS_mf_empirico);